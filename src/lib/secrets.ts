import { 
  SecretsManagerClient, 
  GetSecretValueCommand 
} from '@aws-sdk/client-secrets-manager';

interface AppSecrets {
  cohereApiKey: string;
  tavilyApiKey: string;
  langsmithApiKey: string;
  bedrockApiKey: string;
  redisUrl: string;
  redisPassword: string;
  awsAccessKeyId: string;
  awsSecretAccessKey: string;
}

const client = new SecretsManagerClient({ 
  region: process.env.AWS_REGION || 'ap-south-1' 
});

async function loadSecret(secretName: string): Promise<string> {
  const command = new GetSecretValueCommand({ SecretId: secretName });
  const response = await client.send(command);
  return response.SecretString || '';
}

export async function loadAllSecrets(): Promise<AppSecrets> {
  if (process.env.NODE_ENV !== 'production') {
    // Development: use .env file
    return {
      cohereApiKey: process.env.COHERE_API_KEY || '',
      tavilyApiKey: process.env.TAVILY_API_KEY || '',
      langsmithApiKey: process.env.LANGSMITH_API_KEY || '',
      bedrockApiKey: process.env.BEDROCK_API_KEY || '',
      redisUrl: process.env.REDIS_URL || '',
      redisPassword: process.env.REDIS_PASSWORD || '',
      awsAccessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
      awsSecretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
    };
  }

  // Production: load from AWS Secrets Manager
  const [
    cohereSecret,
    tavilySecret,
    langsmithSecret,
    bedrockSecret,
    redisSecret,
    awsCredsSecret
  ] = await Promise.all([
    loadSecret('prod/tutor-agent/cohere-apiKey'),
    loadSecret('prod/tutor-agent/tavily-api-key'),
    loadSecret('prod/tutor-agent/langsmith-api-key'),
    loadSecret('prod/tutor-agent/bedrock-api-key'),
    loadSecret('prod/tutor-agent/redis-cretdentials'),
    loadSecret('prod/tutor-agent/aws-credentials'),
  ]);

  // Parse JSON secrets
  const cohere = JSON.parse(cohereSecret);
  const tavily = JSON.parse(tavilySecret);
  const langsmith = JSON.parse(langsmithSecret);
  const redis = JSON.parse(redisSecret);
  const awsCreds = JSON.parse(awsCredsSecret);

  return {
    cohereApiKey: cohere.COHERE_API_KEY,
    tavilyApiKey: tavily.TAVILY_API_KEY,
    langsmithApiKey: langsmith.LANGSMITH_API_KEY,
    bedrockApiKey: bedrockSecret, // Plain text, no parsing needed
    redisUrl: redis.REDIS_URL,
    redisPassword: redis.REDIS_PASSWORD,
    awsAccessKeyId: awsCreds.AWS_ACCESS_KEY_ID,
    awsSecretAccessKey: awsCreds.AWS_SECRET_ACCESS_KEY,
  };
}
