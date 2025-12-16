import { render } from '@testing-library/react';
import Hero from '.';

describe('Hero', () => {
  test('renders its content', () => {
    const { container } = render(<Hero />);

    expect(container).not.toBeEmptyDOMElement();
  });
});
