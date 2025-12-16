import { render } from '@testing-library/react';
import Nav from '.';

describe('Nav', () => {
  test('renders its content', () => {
    const { container } = render(<Nav />);

    expect(container).not.toBeEmptyDOMElement();
  });
});
