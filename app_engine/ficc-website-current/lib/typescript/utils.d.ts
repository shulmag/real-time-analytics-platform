declare type Maybe<T> = T | undefined | null;
declare type _string = Maybe<string>;
declare type _number = Maybe<number>;
declare type R<T> = Readonly<T>;
declare type RA<T> = ReadonlyArray<T>;
