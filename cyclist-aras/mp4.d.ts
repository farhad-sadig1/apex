declare module '*.mp4' {
  /** Metro asset ID on native; URL string or `{ uri }` record on some web bundlers. */
  const src: number | string | { uri: string };
  export default src;
}
