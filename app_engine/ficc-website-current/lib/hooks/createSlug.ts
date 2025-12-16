type CreateSlug = (x?: string | null) => string | undefined;


export const createSlug: CreateSlug = arg => (
  arg
    ?.replace(/\s|_/g, '-')
    .replace(/\./g, '')
    .toLocaleLowerCase()
);
