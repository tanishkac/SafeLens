import { z } from "zod";

// Optional environment variables with fallback defaults
const EnvSchema = z.object({
  BACKEND_URL: z.string().optional().default("http://localhost:8000"),
  AUTH_SECRET: z.string().optional().default("safelens-default-local-secret-32charsmin"),
});

const parsed = EnvSchema.safeParse({
  BACKEND_URL: process.env.BACKEND_URL,
  AUTH_SECRET: process.env.AUTH_SECRET,
});

export const env = {
  BACKEND_URL: (parsed.success ? parsed.data.BACKEND_URL : process.env.BACKEND_URL) || "http://localhost:8000",
  AUTH_SECRET: (parsed.success ? parsed.data.AUTH_SECRET : process.env.AUTH_SECRET) || "safelens-default-local-secret-32charsmin",
  AUTH_PROVIDER_ID: "local",
  AUTH_PROVIDER_NAME: "Local",
  AUTH_CLIENT_ID: "local",
  AUTH_CLIENT_SECRET: "local",
  AUTH_ISSUER_URL: "http://localhost:3000",
};

export const AUTH_ISSUER_ORIGIN = "http://localhost:3000";
export const OIDC_WELL_KNOWN = "http://localhost:3000/.well-known/openid-configuration";


