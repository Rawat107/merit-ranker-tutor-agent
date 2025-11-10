import {  AmazonKnowledgeBaseRetriever } from '@langchain/aws'
import pino from 'pino';
import { Document } from '../types/index.js';

export class AWSKnowledgeBaseRetriever {
  private retriever: AmazonKnowledgeBaseRetriever;

  constructor(private logger: pino.Logger){
    this.logger = logger;

    const kbId = process.env.KB_ID;
    const region = process.env.AWS_REGION || 'ap-southeast-2';

    if (!kbId) {
      this.logger.error('Knowledge Base ID (KB_ID) is not set in environment variables.');
      throw new Error('Knowledge Base ID (KB_ID) is required');
    }

    this.logger.info({ kbId, region }, 'Initializing AWS Knowledge Base Retriever');

    //Initialize the AmazonKnowledgeBaseRetriever
    this.retriever = new AmazonKnowledgeBaseRetriever({
      knowledgeBaseId: kbId,
      region,
      topK: 5,    
    });

    this.logger.info("AWS knowledge base retriever initialized")
  }

  /**
   * Retrieve relevant documents from Knowledge Base
   * @param query - The search query
   * @param options - Retrieval options (subject, level, k)
   * @returns Array of relevant documents
   */

  async getRelevantDocuments(
    query: string,
    options?: {
      subject?: string;
      level?: string;
      k?: number;
    }
  ): Promise<Document[]>{
     const k = options?.k || 5;


     this.logger.info(
      {
        query: query.substring(0, 100),
        subject: options?.subject,
        level: options?.level,
        k,
      },
      'Retrieving documents from AWS Knowledge Base'
     );

     try {
      const langchainDocs = await this.retriever.invoke(query);

      if(!langchainDocs || langchainDocs.length === 0){
        this.logger.warn('[kB Retriever] No documents found for the query');
        return [];
      }

      //Transform langchain documents to our Document format
      // Transform LangChain Documents to our Document format
      const documents: Document[] = langchainDocs.map((doc, index) => ({
        id: doc.metadata?.id || `kb-doc-${index}`,
        text: doc.pageContent,
        metadata: {
          source: 'knowledge-base',
          subject: options?.subject || 'unknown',
          level: options?.level || 'unknown',
          score: doc.metadata?.score || 0.7,
          ...doc.metadata,
        },
        score: doc.metadata?.score || 0.7,
      }));

      this.logger.info(
        {
          count: documents.length,
          topScore: documents[0]?.score,
          sources: documents.map((d) => d.id),
        },
        '[KBRetriever] Retrieved documents from KB'
      )

      return documents

     } catch (error: any) {
      this.logger.error({ error: error.message, query: query.substring(0,100) }, 
      '[KBRetriever] Error retrieving documents from KB');
      return [];
    }
  }
}