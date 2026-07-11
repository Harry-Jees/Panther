import { setPassword, getPassword, deletePassword } from "cross-keychain";

const SERVICE_NAME = "panther-cli";

/**
 * Provider names that support BYOK credential storage.
 */
export type CredentialProvider = "openai" | "anthropic" | "gemini";

/**
 * Store an API key in the OS-native keychain.
 */
export async function addCredential(provider: CredentialProvider, key: string): Promise<void> {
  await setPassword(SERVICE_NAME, provider, key);
}

/**
 * Retrieve an API key from the OS-native keychain.
 * Returns null if no credential is stored for the provider.
 */
export async function getCredential(provider: CredentialProvider): Promise<string | null> {
  try {
    return await getPassword(SERVICE_NAME, provider);
  } catch {
    return null;
  }
}

/**
 * Remove an API key from the OS-native keychain.
 */
export async function removeCredential(provider: CredentialProvider): Promise<void> {
  await deletePassword(SERVICE_NAME, provider);
}

/**
 * Check if a credential exists for the given provider.
 */
export async function hasCredential(provider: CredentialProvider): Promise<boolean> {
  const cred = await getCredential(provider);
  return cred !== null;
}

/**
 * List all providers that have stored credentials (without revealing the key values).
 */
export async function listCredentials(): Promise<CredentialProvider[]> {
  const providers: CredentialProvider[] = ["openai", "anthropic", "gemini"];
  const result: CredentialProvider[] = [];

  for (const provider of providers) {
    if (await hasCredential(provider)) {
      result.push(provider);
    }
  }

  return result;
}
