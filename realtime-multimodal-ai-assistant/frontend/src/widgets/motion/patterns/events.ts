// ----------------------------------------------------------------------

/**
 * Creates motion variants for hover and tap interactions
 * @param {number} hoverScale - Expansion factor on hover (default 1.09)
 * @param {number} tapScale - Contraction factor on tap (default 0.97)
 * @returns {Object} Motion variants object with hover and tap states
 */
function createInteractionVariants(
  hoverScale: number = 1.1,
  tapScale: number = 0.9
): { hover: { scale: number }; tap: { scale: number } } {
  return {
    hover: {
      scale: hoverScale,
    },
    tap: {
      scale: tapScale,
    },
  };
}

export const varHover = createInteractionVariants;
