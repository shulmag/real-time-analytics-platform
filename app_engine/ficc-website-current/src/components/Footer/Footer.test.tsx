import { render } from '@testing-library/react';
import Footer from '.';

describe('Footer', () => {
  test('renders its content', () => {
    const { container } = render(<Footer />);

    expect(container).not.toBeEmptyDOMElement();
  });
});
