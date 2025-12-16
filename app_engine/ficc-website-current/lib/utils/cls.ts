type Args = Array<string | false | undefined>;

export const cls = (...args: Args): string | undefined => {
  const cl: Args = args.filter(arg => typeof arg === 'string');

  if (cl.length === 0) {
    return undefined;
  }

  return cl.join(' ').trim();
};
