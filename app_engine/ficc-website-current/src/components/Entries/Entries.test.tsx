import { render } from '@testing-library/react';
import Entries from '.';

describe('Entries', () => {
  test('does not render content without "entries"', () => {
    const { container } = render(<Entries />);

    expect(container).toBeEmptyDOMElement();
  });
});
