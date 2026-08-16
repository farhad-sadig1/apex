/**
 * Metro/Expo packaged asset. On native this is a numeric registry ID.
 * On web it may already be a URL string, a `{ uri }` record, or a module ID.
 */
export type PackagedAsset = number | string | { uri: string };

export function isUriRecord(value: unknown): value is { uri: string } {
  return (
    typeof value === 'object' &&
    value !== null &&
    'uri' in value &&
    typeof (value as { uri: unknown }).uri === 'string'
  );
}

export interface AssetUriResolvers {
  fromModule: (moduleId: number) => { uri?: string | null };
  resolveAssetSource: (moduleId: number) => { uri?: string | null } | null | undefined;
}

/**
 * Resolve a bundled file to a playable URI without touching react-native-web's
 * Image (which has no `resolveAssetSource`).
 *
 * JSON telemetry inlined by Metro is an object without `uri` — this returns
 * '' rather than throwing.
 */
export function resolveAssetUri(
  asset: unknown,
  platformOS: string,
  resolvers: AssetUriResolvers,
): string {
  if (typeof asset === 'string') {
    return asset;
  }
  if (isUriRecord(asset)) {
    return asset.uri;
  }
  if (typeof asset !== 'number' || !Number.isFinite(asset)) {
    return '';
  }
  if (platformOS === 'web') {
    return resolvers.fromModule(asset).uri ?? '';
  }
  return resolvers.resolveAssetSource(asset)?.uri ?? '';
}
