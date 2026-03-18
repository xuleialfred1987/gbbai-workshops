/**
 * Polyfill for window.crypto.randomUUID
 * This must be loaded before any other scripts that use crypto.randomUUID
 */
// Use a named IIFE to satisfy func-names rule
(function installCryptoPolyfill() {
    if (window.crypto && !window.crypto.randomUUID) {
      console.log('Installing crypto.randomUUID polyfill');
      
      // Add the randomUUID method directly to the crypto object
      Object.defineProperty(window.crypto, 'randomUUID', {
        configurable: true,
        writable: true,
        // Simplified arrow function without block statement
        value: () => '10000000-1000-4000-8000-100000000000'.replace(/[018]/g, (c) => {
          const randomValue = window.crypto.getRandomValues(new Uint8Array(1))[0];
          // eslint-disable-next-line no-bitwise
          return (c ^ (randomValue & 15) >> (Number(c) / 4)).toString(16);
        })
      });
    }
  })();