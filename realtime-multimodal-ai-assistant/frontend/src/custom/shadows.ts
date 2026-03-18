// mui
import { alpha, Shadows } from '@mui/material/styles';

// project imports
import { grey, common } from './color-set';

// ----------------------------------------------------------------------

export function shadows(mode: 'light' | 'dark'): Shadows {
  const baseColor = mode === 'light' ? grey[500] : common.black;

  const opacityPrimary = 0.2;
  const opacitySecondary = 0.14;
  const opacityTertiary = 0.12;

  const tPrimary = alpha(baseColor, opacityPrimary);
  const tSecondary = alpha(baseColor, opacitySecondary);
  const tTertiary = alpha(baseColor, opacityTertiary);

  const s = (x: number, y: number, z: number, w: number) => `${x}px ${y}px ${z}px ${w}px`;

  const shadowValues = [
    'none',
    `${s(0, 2, 1, -1)} ${tPrimary}, ${s(0, 1, 1, 0)} ${tSecondary}, ${s(0, 1, 3, 0)} ${tTertiary}`,
    `${s(0, 3, 1, -2)} ${tPrimary}, ${s(0, 2, 2, 0)} ${tSecondary}, ${s(0, 1, 5, 0)} ${tTertiary}`,
    `${s(0, 3, 3, -2)} ${tPrimary}, ${s(0, 3, 4, 0)} ${tSecondary}, ${s(0, 1, 8, 0)} ${tTertiary}`,
    `${s(0, 2, 4, -1)} ${tPrimary}, ${s(0, 4, 5, 0)} ${tSecondary}, ${s(0, 1, 10, 0)} ${tTertiary}`,
    `${s(0, 3, 5, -1)} ${tPrimary}, ${s(0, 5, 8, 0)} ${tSecondary}, ${s(0, 1, 14, 0)} ${tTertiary}`,
    `${s(0, 3, 5, -1)} ${tPrimary}, ${s(0, 6, 10, 0)} ${tSecondary}, ${s(0, 1, 18, 0)} ${tTertiary}`,
    `${s(0, 4, 5, -2)} ${tPrimary}, ${s(0, 7, 10, 1)} ${tSecondary}, ${s(0, 2, 16, 1)} ${tTertiary}`,
    `${s(0, 5, 5, -3)} ${tPrimary}, ${s(0, 8, 10, 1)} ${tSecondary}, ${s(0, 3, 14, 2)} ${tTertiary}`,
    `${s(0, 5, 6, -3)} ${tPrimary}, ${s(0, 9, 12, 1)} ${tSecondary}, ${s(0, 3, 16, 2)} ${tTertiary}`,
    `${s(0, 6, 6, -3)} ${tPrimary}, ${s(0, 10, 14, 1)} ${tSecondary}, ${s(0, 4, 18, 3)} ${tTertiary}`,
    `${s(0, 6, 7, -4)} ${tPrimary}, ${s(0, 11, 15, 1)} ${tSecondary}, ${s(0, 4, 20, 3)} ${tTertiary}`,
    `${s(0, 7, 8, -4)} ${tPrimary}, ${s(0, 12, 17, 2)} ${tSecondary}, ${s(0, 5, 22, 4)} ${tTertiary}`,
    `${s(0, 7, 8, -4)} ${tPrimary}, ${s(0, 13, 19, 2)} ${tSecondary}, ${s(0, 5, 24, 4)} ${tTertiary}`,
    `${s(0, 7, 9, -4)} ${tPrimary}, ${s(0, 14, 21, 2)} ${tSecondary}, ${s(0, 5, 26, 4)} ${tTertiary}`,
    `${s(0, 8, 9, -5)} ${tPrimary}, ${s(0, 15, 22, 2)} ${tSecondary}, ${s(0, 6, 28, 5)} ${tTertiary}`,
    `${s(0, 8, 10, -5)} ${tPrimary}, ${s(0, 16, 24, 2)} ${tSecondary}, ${s(0, 6, 30, 5)} ${tTertiary}`,
    `${s(0, 8, 11, -5)} ${tPrimary}, ${s(0, 17, 26, 2)} ${tSecondary}, ${s(0, 6, 32, 5)} ${tTertiary}`,
    `${s(0, 9, 11, -5)} ${tPrimary}, ${s(0, 18, 28, 2)} ${tSecondary}, ${s(0, 7, 34, 6)} ${tTertiary}`,
    `${s(0, 9, 12, -6)} ${tPrimary}, ${s(0, 19, 29, 2)} ${tSecondary}, ${s(0, 7, 36, 6)} ${tTertiary}`,
    `${s(0, 10, 13, -6)} ${tPrimary}, ${s(0, 20, 31, 3)} ${tSecondary}, ${s(0, 8, 38, 7)} ${tTertiary}`,
    `${s(0, 10, 13, -6)} ${tPrimary}, ${s(0, 21, 33, 3)} ${tSecondary}, ${s(0, 8, 40, 7)} ${tTertiary}`,
    `${s(0, 10, 14, -6)} ${tPrimary}, ${s(0, 22, 35, 3)} ${tSecondary}, ${s(0, 8, 42, 7)} ${tTertiary}`,
    `${s(0, 11, 14, -7)} ${tPrimary}, ${s(0, 23, 36, 3)} ${tSecondary}, ${s(0, 9, 44, 8)} ${tTertiary}`,
    `${s(0, 11, 15, -7)} ${tPrimary}, ${s(0, 24, 38, 3)} ${tSecondary}, ${s(0, 9, 46, 8)} ${tTertiary}`,
  ];

  return shadowValues as Shadows;
}
