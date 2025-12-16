import {
  FC,
  useEffect,
  useRef,
} from 'react';
import {
  disableBodyScroll,
  enableBodyScroll,
  clearAllBodyScrollLocks,
} from 'body-scroll-lock';
import { state } from '@providers/StateProvider';
import { createSlug } from '#hooks';
import styles from './Nav.module.scss';

import type { NavProps as Props } from './types';


const Nav: FC<Props> = ({ links }) => {
  const $nav = useRef<HTMLElement>(null);
  const $menu = useRef<HTMLDivElement>(null);
  const _gate = useRef(false);
  const { slug } = state();

  const toggleMenu = (): void => {
    if (window.innerWidth > 1024) {
      return;
    }

    const menu = $menu.current;

    if (menu) {
      window.requestAnimationFrame(() => {
        menu.classList.toggle(styles.open);

        if (menu.classList.contains(styles.open)) {
          disableBodyScroll(menu);
        } else {
          enableBodyScroll(menu);
        }
      });
    }
  };

  useEffect(() => {
    const toggleScroll = (): void => {
      const nav = $nav.current;

      if (nav) {
        if (window.pageYOffset >= 10 && !_gate.current) {
          nav.classList.add(styles.scroll);

          _gate.current = true;
        }

        if (window.pageYOffset < 10 && _gate.current) {
          nav.classList.remove(styles.scroll);

          _gate.current = false;
        }
      }
    };

    window.addEventListener('scroll', toggleScroll, { passive: true });

    return () => {
      window.removeEventListener('scroll', toggleScroll);
      clearAllBodyScrollLocks();
    };
  }, []);

  return (
    <nav
      className={styles.root}
      ref={$nav}
    >
      <div
        className={styles.wrapper}
        data-wrap
      >
        <a
          className={styles.logo}
          href={slug === '/' ? '#' : '/'}
        >
          ficc<span>.ai</span>
        </a>
        <div
          className={styles.menu}
          ref={$menu}
        >
          {links?.length && links.map(item => {
            if (item.toLowerCase().includes('login')) {
              return (
                <a
                  href="https://pricing.ficc.ai/"
                  key={createSlug(item)}
                  onClick={toggleMenu}
                >
                  {item}
                </a>
              );
            }

            return (
              <a
                href={`/#${createSlug(item)}`}
                key={createSlug(item)}
                onClick={toggleMenu}
              >
                {item}
              </a>
            );
          })}
          <p aria-hidden="true">ficc<span>.ai</span></p>
          <button
            className={`${styles.icon} ${styles.close} graphical`}
            type="button"
            onClick={toggleMenu}
          >
            <span>Close nav menu</span>
            <svg
              aria-hidden="true"
              viewBox="0 0 40 40"
            >
              <path d="M10,10 30,30" />
              <path d="M30,10 10,30" />
            </svg>
          </button>
        </div>
        <button
          className={`${styles.icon} graphical`}
          type="button"
          onClick={toggleMenu}
        >
          <span>Open nav menu</span>
          <svg
            aria-hidden="true"
            viewBox="0 0 40 40"
          >
            <path d="M8,12h22" />
            <path d="M8,20h10" />
            <path d="M8,28h15" />
          </svg>
        </button>
      </div>
    </nav>
  );
};


export default Nav;
