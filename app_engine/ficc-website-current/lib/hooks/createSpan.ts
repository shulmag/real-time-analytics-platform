import { underlineRegEx } from '#constants';


type CreateSpan = (x?: string | null) => string;


export const createSpan: CreateSpan = arg => (
  arg?.replace(underlineRegEx, '<span>$1</span>') || `${arg}`
);
