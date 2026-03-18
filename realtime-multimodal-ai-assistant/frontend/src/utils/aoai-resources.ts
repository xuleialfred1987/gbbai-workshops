import { getStorage, AOAI_CREDENTIAL_KEY } from 'src/hooks/local-storage';

import { IAoaiResourceItem } from 'src/types/azure-resource';

// ----------------------------------------------------------------------

export default function getInitialResourceName(models?: string[]) {
  const aoaiCredentials: IAoaiResourceItem[] = getStorage(AOAI_CREDENTIAL_KEY) ?? [];

  // 1) optionally filter by models (substring match, OR‑ed)
  const filtered =
    models && models.length
      ? aoaiCredentials.filter((item) => models.some((m) => item.model?.includes(m)))
      : aoaiCredentials;

  // fallback to full list if the filter produced nothing
  const candidates = filtered.length ? filtered : aoaiCredentials;

  // 2) prefer resources marked as primary
  const primary = candidates.find((item) => item.primary);
  if (primary) return primary.resourceName;

  // 3) otherwise choose the first non‑embedding model, if any
  const nonEmbedding = candidates.find((item) => !item.model.includes('embedding'));
  if (nonEmbedding) return nonEmbedding.resourceName;

  // 4) final fallback: first candidate or empty string
  return candidates[0]?.resourceName ?? '';
}

/**
 * Return all matching resource names.
 */
export function getResourceNames(models?: string[]) {
  const aoaiCredentials: IAoaiResourceItem[] = getStorage(AOAI_CREDENTIAL_KEY) ?? [];

  const filtered =
    models && models.length
      ? aoaiCredentials.filter((item) => models.some((m) => item.model?.includes(m)))
      : aoaiCredentials;

  return filtered.map((item) => item.resourceName);
}
