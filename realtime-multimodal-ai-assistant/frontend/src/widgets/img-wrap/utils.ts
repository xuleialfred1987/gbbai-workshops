type AspectRatio = [string, string];

interface RatioConfig {
  [key: string]: AspectRatio;
}

interface RatioCalculations {
  [key: string]: string;
}

const ASPECT_RATIOS: RatioConfig = {
  standard: ['4', '3'],
  portrait: ['3', '4'],
  wide: ['6', '4'],
  tall: ['4', '6'],
  widescreen: ['16', '9'],
  vertical: ['9', '16'],
  ultrawide: ['18', '9'],
  ultratall: ['9', '18'],
  cinematic: ['21', '9'],
  supervertical: ['9', '21'],
  square: ['1', '1'],
};

const calculateAspectRatio = (numerator: string, denominator: string): string =>
  denominator === '1' ? '100%' : `calc(100% / ${numerator} * ${denominator})`;

const RATIO_CALCULATIONS: RatioCalculations = new Proxy(
  {},
  {
    get(_target: {}, key: string): string {
      const [num, den] = ASPECT_RATIOS[key] || ASPECT_RATIOS.square;
      return calculateAspectRatio(num, den);
    },
  }
);

export function calculateRatio(ratio = '1/1'): string {
  const [num, den] = ratio.split('/');

  const matchingRatio = Object.entries(ASPECT_RATIOS).find(
    ([_, [aspNum, aspDen]]) => aspNum === num && aspDen === den
  );

  return matchingRatio ? RATIO_CALCULATIONS[matchingRatio[0]] : RATIO_CALCULATIONS.square;
}
