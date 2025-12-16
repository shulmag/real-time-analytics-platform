export const isType = <T>(arg: T, prop: keyof T): arg is T => prop in arg;

export const isElement = (arg: unknown): arg is JSX.Element => (
  typeof (arg as JSX.Element).type === 'function'
);

export const isEven = (arg?: number): boolean => {
  if (!arg) {
    return false;
  }

  return arg % 2 === 0;
};

export const isOdd = (arg?: number): boolean => {
  if (!arg) {
    return false;
  }

  return arg % 2 !== 0;
};
