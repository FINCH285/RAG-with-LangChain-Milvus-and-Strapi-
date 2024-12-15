import express from "express";
import axios from "axios";
import dotenv from "dotenv";
import cors from "cors";
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Milvus } from "@langchain/community/vectorstores/milvus";
import { Document } from "langchain/document";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

dotenv.config();

const CONFIG = {
    PORT: parseInt(process.env.PORT || "30080", 10),
    STRAPI: {
        URL: "http://localhost:1337/api/milvus-knowledgebases",
        TIMEOUT: 5000
    },
    MILVUS: {
        URL: "localhost:19530",
        COLLECTION: "rag_collection",
        PRIMARY_FIELD: "pk",
        VECTOR_FIELD: "vector",
        TEXT_FIELD: "text",
        TEXT_MAX_LENGTH: 4096,
        SEARCH_PARAMS: {
            nprobe: 16,
            offset: 0
        },
        BATCH_SIZE: 100
    },
    CHUNKING: {
        SIZE: 2000,
        OVERLAP: 200
    },
    TOP_K: 3
};

// Initialize services
const app = express();
app.use(express.json({ limit: '1mb' }));
app.use(cors());

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    openAIApiKey: process.env.OPENAI_API_KEY,
    maxRetries: 3,
    timeout: 30000,
});

const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    maxRetries: 3,
});

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CONFIG.CHUNKING.SIZE,
    chunkOverlap: CONFIG.CHUNKING.OVERLAP,
    separators: ["\n\n", "\n", " ", ""]
});

// Milvus store singleton
let milvusStore = null;
let lastDataHash = null;

const processDocument = (content) => {
    if (!content?.Content || !content.Title) return null;
    
    const processedContent = content.Content
        .map(section => {
            if (section.type === 'paragraph' && section.children) {
                return section.children
                    .map(child => child.text)
                    .join(' ')
                    .trim();
            }
            return null;
        })
        .filter(Boolean)
        .join('\n\n');

    return processedContent ? new Document({
        pageContent: processedContent,
        metadata: { 
            source: 'milvus_content', 
            id: content.id,
            title: content.Title,
            documentId: content.documentId
        }
    }) : null;
};

// Initialize or get Milvus store
async function getMilvusStore() {
    if (milvusStore) {
        return milvusStore;
    }

    try {
        // First try to connect to existing collection
        try {
            console.log('Attempting to connect to existing collection');
            milvusStore = await Milvus.fromExistingCollection(
                embeddings,
                {
                    url: CONFIG.MILVUS.URL,
                    collectionName: CONFIG.MILVUS.COLLECTION,
                    primaryField: CONFIG.MILVUS.PRIMARY_FIELD,
                    vectorField: CONFIG.MILVUS.VECTOR_FIELD,
                    textField: CONFIG.MILVUS.TEXT_FIELD,
                    textFieldMaxLength: CONFIG.MILVUS.TEXT_MAX_LENGTH,
                }
            );
            return milvusStore;
        } catch (error) {
            console.log('Collection does not exist, will create new one');
            // If collection doesn't exist, fetch data and create it
            const response = await axios.get(CONFIG.STRAPI.URL, {
                timeout: CONFIG.STRAPI.TIMEOUT
            });
            
            const processPromises = response.data.data.map(processDocument);
            const processedDocs = await Promise.all(processPromises);
            
            const uniqueDocs = new Map();
            processedDocs
                .filter(Boolean)
                .forEach(doc => uniqueDocs.set(doc.metadata.documentId, doc));

            const docs = Array.from(uniqueDocs.values());
            
            // Split documents
            const splitPromises = docs.map(doc => splitter.splitDocuments([doc]));
            const splitDocs = await Promise.all(splitPromises);
            const allSplitDocs = splitDocs.flat();

            console.log(`Creating new collection with ${allSplitDocs.length} documents`);
            
            milvusStore = await Milvus.fromDocuments(
                allSplitDocs,
                embeddings,
                {
                    url: CONFIG.MILVUS.URL,
                    collectionName: CONFIG.MILVUS.COLLECTION,
                    primaryField: CONFIG.MILVUS.PRIMARY_FIELD,
                    vectorField: CONFIG.MILVUS.VECTOR_FIELD,
                    textField: CONFIG.MILVUS.TEXT_FIELD,
                    textFieldMaxLength: CONFIG.MILVUS.TEXT_MAX_LENGTH,
                }
            );

            lastDataHash = Buffer.from(JSON.stringify(response.data)).toString('base64');
            return milvusStore;
        }
    } catch (error) {
        console.error('Failed to initialize Milvus:', error);
        throw error;
    }
}

// Update Milvus data if needed
async function updateMilvusData() {
    try {
        const response = await axios.get(CONFIG.STRAPI.URL, {
            timeout: CONFIG.STRAPI.TIMEOUT
        });
        
        const currentHash = Buffer.from(JSON.stringify(response.data)).toString('base64');
        if (currentHash === lastDataHash) {
            console.log('Data unchanged, skipping update');
            return false;
        }

        console.log('Content changed, updating Milvus collection');

        const processPromises = response.data.data.map(processDocument);
        const processedDocs = await Promise.all(processPromises);
        
        const uniqueDocs = new Map();
        processedDocs
            .filter(Boolean)
            .forEach(doc => uniqueDocs.set(doc.metadata.documentId, doc));

        const docs = Array.from(uniqueDocs.values());
        
        // Split documents
        const splitPromises = docs.map(doc => splitter.splitDocuments([doc]));
        const splitDocs = await Promise.all(splitPromises);
        const allSplitDocs = splitDocs.flat();

        // Delete all existing documents before adding new ones
        await milvusStore.delete({});
        
        // Add new documents in batches
        for (let i = 0; i < allSplitDocs.length; i += CONFIG.MILVUS.BATCH_SIZE) {
            const batch = allSplitDocs.slice(i, i + CONFIG.MILVUS.BATCH_SIZE);
            await milvusStore.addDocuments(batch);
            console.log(`Added batch ${Math.floor(i / CONFIG.MILVUS.BATCH_SIZE) + 1} of ${Math.ceil(allSplitDocs.length / CONFIG.MILVUS.BATCH_SIZE)}`);
        }

        lastDataHash = currentHash;
        console.log(`Updated Milvus with ${allSplitDocs.length} documents`);
        return true;
    } catch (error) {
        console.error("Error updating Milvus data:", error);
        throw error;
    }
}

async function handleQuery(chatHistory, input) {
    const store = await getMilvusStore();
    
    try {
        await updateMilvusData();
    } catch (error) {
        console.warn("Failed to check for updates:", error);
    }
    
    const results = await store.similaritySearchWithScore(
        input, 
        CONFIG.TOP_K
    );
    
    console.log('\n=== Retrieved Documents from Milvus ===');
    results.forEach(([doc, score], index) => {
        console.log(`\nDocument ${index + 1} (score: ${score}):`);
        console.log('Title:', doc.metadata.title);
        console.log('Content:', doc.pageContent.substring(0, 150) + '...');
    });

    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt: ChatPromptTemplate.fromMessages([
            [
                "system",
                `You are an AI assistant specializing in Milvus and Zilliz, vector database technologies. Your goal is to provide accurate and helpful answers based on the provided context.
    
    **Guidelines:**
    1. **Milvus/Zilliz-Related Queries:**
       - If the query is about Milvus or Zilliz and the context includes relevant information, provide a detailed and structured response.
       - Use proper Markdown formatting:
         - \`\`\` for code blocks
         - **Bold** for emphasis
         - Bullet points for lists
         - ### Headings for sections
       - If no relevant information exists in the context, respond with: *"I don't have that information yet."*
    
    2. **Unrelated Queries:**
       - If the query is not about Milvus or Zilliz, respond with: 
         *"This topic is outside my expertise. I specialize in Milvus and Zilliz. Please ask questions related to these technologies."*
    
    **Context:**
    {context}
    
    **User Query:**
    {input}`
            ],
            ...chatHistory,
            ["user", "{input}"]
        ]),
        documentPrompt: ChatPromptTemplate.fromTemplate("Content: {page_content}\n\n")
    });
    
    
    const response = await chain.invoke({
        input,
        context: results.map(([doc]) => doc)
    });

    return { 
        answer: response,
        context: results.map(([doc]) => ({
            content: doc.pageContent,
            title: doc.metadata.title,
            id: doc.metadata.id,
            documentId: doc.metadata.documentId
        }))
    };
}

// API endpoints
app.post("/chat", async (req, res) => {
    try {
        const { chatHistory, input } = req.body;
        
        if (!input?.trim()) {
            return res.status(400).json({ 
                error: "Invalid input",
                message: "No input provided" 
            });
        }

        const formattedHistory = Array.isArray(chatHistory)
            ? chatHistory.map(msg => 
                msg.role === "user" 
                    ? new HumanMessage(msg.content)
                    : new AIMessage(msg.content)
            )
            : [];

        const response = await handleQuery(formattedHistory, input);
        res.json(response);
    } catch (error) {
        console.error("Chat request error:", error);
        res.status(500).json({
            error: "Internal server error",
            message: process.env.NODE_ENV === 'production' 
                ? "An unexpected error occurred"
                : error.message
        });
    }
});

app.get("/health", async (req, res) => {
    try {
        const store = await getMilvusStore();
        res.json({
            status: "ok",
            timestamp: new Date().toISOString(),
            milvusInitialized: !!store,
            config: CONFIG
        });
    } catch (error) {
        res.status(500).json({
            status: "error",
            error: error.message
        });
    }
});

// Start server
app.listen(CONFIG.PORT, () => {
    console.log(`Server running on http://localhost:${CONFIG.PORT}`);
    console.log('Configuration:', CONFIG);
});