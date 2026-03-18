import { useState } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

import { fCurrency } from 'src/utils/number-formatter';

import { bgBlur } from 'src/custom/css';

import Image from 'src/widgets/img-wrap';
import Iconify from 'src/widgets/iconify';

import IncrementerButton from './function-call-handlers-incrementer-button';

// ----------------------------------------------------------------------

type Props = {
  data: any;
};

export default function FunctionAddToCartHandler({ data }: Props) {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const { product, price } = data;

  const [quantity, setQuantity] = useState(1);
  const [selectedColor, setSelectedColor] = useState('primary.main');

  // Define color options
  const colorOptions = [
    { value: 'primary.main', label: 'Blue' },
    { value: '#AFEDD6', label: 'Green' },
    { value: '#9FA5B4', label: 'Silver' },
  ];

  // Define the maximum available quantity
  const available = 10;

  // Handler functions for incrementer
  const onIncrease = () => {
    if (quantity < available) {
      setQuantity((prevQuantity) => prevQuantity + 1);
    }
  };

  const onDecrease = () => {
    if (quantity > 1) {
      setQuantity((prevQuantity) => prevQuantity - 1);
    }
  };

  return (
    <Stack
      spacing={3.5}
      alignItems="center"
      justifyContent="space-between"
      sx={{
        p: 3,
        pt: 2.5,
        px: 3,
        mt: 2,
        width: 0.5,
        minWidth: '300px',
        bgcolor: isDarkMode
          ? alpha(theme.palette.grey[800], 0.6)
          : alpha(theme.palette.grey[100], 0.8),
        border: `1px solid ${alpha(
          isDarkMode ? theme.palette.grey[700] : theme.palette.grey[300],
          0.5
        )}`,
        borderRadius: 1.75,
        color: isDarkMode ? 'common.white' : 'text.primary',
        ...bgBlur({
          color: isDarkMode ? theme.palette.grey[800] : theme.palette.common.white,
          opacity: isDarkMode ? 0.44 : 0.7,
        }),
        transition: theme.transitions.create(['background-color', 'border-color'], {
          duration: theme.transitions.duration.shorter,
        }),
      }}
    >
      <Stack alignItems="center" spacing={1.25}>
        <Iconify icon="solar:cart-plus-bold" width={32} sx={{ color: 'primary.main' }} />
        <Typography variant="h6" color={isDarkMode ? 'grey.400' : 'grey.700'}>
          Add to cart
        </Typography>
      </Stack>

      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        spacing={0}
        sx={{ width: 1, mt: 1 }}
      >
        <Stack direction="row" alignItems="center" spacing={1.5}>
          <Image
            alt={product}
            src="/assets/images/smart-phone/snapshot1.jpeg"
            sx={{ width: 64, height: 64, borderRadius: 1.5 }}
          />
          <Stack alignItems="flex-start" justifyContent="flex-start" spacing={2}>
            <Typography
              variant="subtitle2"
              textTransform="capitalize"
              noWrap
              sx={{
                maxWidth: '140px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                width: '100%', // Ensures the container takes available space up to maxWidth
              }}
            >
              {product}
            </Typography>

            <Stack direction="row" spacing={1.5} alignItems="center">
              {colorOptions.map((color) => (
                <Box
                  key={color.value}
                  onClick={() => setSelectedColor(color.value)}
                  sx={{
                    width: 14,
                    height: 14,
                    cursor: 'pointer',
                    bgcolor: color.value,
                    borderRadius: '50%',
                    border:
                      selectedColor === color.value
                        ? `1.5px solid ${isDarkMode ? theme.palette.common.white : theme.palette.grey[800]}`
                        : 'none',
                    boxShadow: `inset -1px 1px 2px ${alpha(theme.palette.common.black, 0.24)}`,
                    position: 'relative',
                    '&:hover': {
                      opacity: 0.8,
                    },
                  }}
                />
              ))}
            </Stack>
          </Stack>
        </Stack>

        <Stack alignItems="center" spacing={1.25}>
          <IncrementerButton
            quantity={quantity}
            onDecrease={onDecrease}
            onIncrease={onIncrease}
            disabledDecrease={quantity <= 1}
            disabledIncrease={quantity >= available}
          />
          <Typography variant="subtitle2">{fCurrency(price * quantity)} </Typography>
        </Stack>
      </Stack>

      {/* Action Buttons */}
      <Stack direction="row" spacing={2} sx={{ width: '100%', mt: 2 }}>
        <Button
          variant="outlined"
          fullWidth
          sx={{
            color: isDarkMode ? theme.palette.grey[300] : theme.palette.grey[700],
            borderColor: isDarkMode ? theme.palette.grey[500] : theme.palette.grey[400],
            '&:hover': {
              borderColor: isDarkMode ? theme.palette.grey[400] : theme.palette.grey[500],
              backgroundColor: isDarkMode
                ? alpha(theme.palette.grey[700], 0.08)
                : alpha(theme.palette.grey[300], 0.12),
            },
          }}
        >
          Cancel
        </Button>

        <Button
          variant="contained"
          fullWidth
          startIcon={<Iconify icon="solar:verified-check-bold" width={16} />}
          sx={{ bgcolor: 'primary.main', '&:hover': { bgcolor: 'primary.dark' } }}
        >
          Ok
        </Button>
      </Stack>
    </Stack>
  );
}
