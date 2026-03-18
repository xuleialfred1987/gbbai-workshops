import * as Yup from 'yup';
import { useSnackbar } from 'notistack';
import { useState, useEffect } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';

import Box from '@mui/material/Box';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import { alpha } from '@mui/material/styles';
import Typography from '@mui/material/Typography';
import LoadingButton from '@mui/lab/LoadingButton';
import DialogActions from '@mui/material/DialogActions';
import { Stack, Dialog, CardHeader } from '@mui/material';

import Iconify from 'src/widgets/iconify';
import FormProvider, { FormAutocomplete } from 'src/widgets/tailored-form';

import { VoiceOption, SpeechOption, SOURCE_LANG_VOICES, SOURCE_LANG_COUNTRIES } from './langs';

// ----------------------------------------------------------------------

const LANGS = SOURCE_LANG_COUNTRIES.map((lang) => lang.name);

// ----------------------------------------------------------------------

type Props = {
  open: boolean;
  onClose: VoidFunction;
  storedConfig: SpeechOption | null;
  onSetSpeechOption?: (config: SpeechOption) => void;
};

export default function VoiceConfigDialog({
  open,
  onClose,
  storedConfig,
  onSetSpeechOption,
}: Props) {
  const { enqueueSnackbar } = useSnackbar();

  const schema = Yup.object().shape({
    language: Yup.string().required('Language is required'),
    voice: Yup.string().required('Voice is required'),
    style: Yup.string().required('Style is required'),
  });

  const defaultValues: { language: string; voice: string; style: string } = storedConfig
    ? {
        language:
          SOURCE_LANG_COUNTRIES.find((lang) => lang.value === storedConfig.lang)?.name ||
          'Chinese Simplified',
        voice: storedConfig.voice,
        style: storedConfig.style.charAt(0).toUpperCase() + storedConfig.style.slice(1),
      }
    : {
        language: 'Chinese Simplified',
        voice: 'zh-CN-XiaoxiaoNeural',
        style: 'Default',
      };

  const methods = useForm({
    resolver: yupResolver(schema),
    defaultValues,
  });

  const {
    control,
    reset,
    handleSubmit,
    formState: { isSubmitting },
  } = methods;

  const [voiceOptions, setVoiceOptions] = useState<VoiceOption[]>([]);
  const [selectedVoiceOption, setSelectedVoiceOption] = useState<VoiceOption>();
  const [styleOptions, setStyleOptions] = useState<string[]>([]);

  const lang = useWatch({
    control,
    name: 'language',
    defaultValue: defaultValues.language,
  });

  useEffect(() => {
    const options = SOURCE_LANG_VOICES[lang] || [];
    setVoiceOptions(options);

    const defaultVoice = defaultValues.voice;
    const matchedOption = options.find((option) => option.value === defaultVoice);

    if (matchedOption) {
      methods.setValue('voice', matchedOption.value);
      setSelectedVoiceOption(matchedOption);
    } else if (options.length > 0) {
      methods.setValue('voice', options[0].value);
      setSelectedVoiceOption(options[0]);
    } else {
      methods.setValue('voice', '');
      setSelectedVoiceOption(undefined);
    }
  }, [lang, methods, defaultValues.voice]);

  useEffect(() => {
    if (selectedVoiceOption) {
      const styles = selectedVoiceOption.styles || [];
      setStyleOptions(styles);

      const defaultStyle = defaultValues.style;
      if (styles.includes(defaultStyle)) {
        methods.setValue('style', defaultStyle);
      } else if (styles.length > 0) {
        methods.setValue('style', styles[0]);
      } else {
        methods.setValue('style', '');
      }
    } else {
      setStyleOptions([]);
      methods.setValue('style', '');
    }
  }, [selectedVoiceOption, methods, defaultValues.style]);

  useEffect(() => {
    if (open) {
      reset(defaultValues);
    }
    // eslint-disable-next-line
  }, [open, reset]);

  const onSubmit = handleSubmit(async (data) => {
    try {
      // console.log(data);
      onSetSpeechOption?.({
        lang: SOURCE_LANG_COUNTRIES.find((_lang) => _lang.name === data.language)?.value || '',
        voice: selectedVoiceOption?.value || '',
        style: data.style.toLowerCase(),
      });
      onClose();
    } catch (_error) {
      enqueueSnackbar(_error, { variant: 'error' });
    }
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      sx={{
        margin: 'auto',
        maxWidth: { xs: '100vw', md: '50%' },
        maxHeight: { xs: '100vh', md: '70%' },
      }}
    >
      <Stack
        sx={{ width: { xs: 'calc(100vw - 60px)', md: 488 }, height: { xs: 'auto', md: 'auto' } }}
      >
        <CardHeader title="Speech configuration" sx={{ mt: -1, mb: 0.5 }} />

        <FormProvider methods={methods} onSubmit={onSubmit}>
          <Box sx={{ mx: 3, mt: 2, mb: 'auto', height: 'auto' }}>
            <Stack sx={{ mt: 1, mb: 3.5 }} spacing={1.75}>
              <Typography variant="subtitle2">Language</Typography>
              <FormAutocomplete
                size="small"
                name="language"
                placeholder="Select language"
                freeSolo
                autoSelect
                options={LANGS}
                getOptionLabel={(option) => option}
                renderOption={(props, option) => (
                  <li {...props} key={option}>
                    {option}
                  </li>
                )}
              />
            </Stack>

            <Stack sx={{ mt: 1, mb: 3 }} spacing={1}>
              <Typography variant="subtitle2">Voice</Typography>
              <FormAutocomplete
                size="small"
                name="voice"
                placeholder="Select voice"
                freeSolo
                autoSelect
                options={voiceOptions}
                getOptionLabel={(option) => (typeof option === 'string' ? option : option.name)}
                isOptionEqualToValue={(option, value) => {
                  const optionName = typeof option === 'string' ? option : option.value;
                  const valueName = typeof value === 'string' ? value : value.value;
                  return optionName === valueName;
                }}
                value={selectedVoiceOption}
                onChange={(event, value) => {
                  if (Array.isArray(value)) {
                    methods.setValue(
                      'voice',
                      value.map((v) => (typeof v === 'string' ? v : v.value)).join(', ')
                    );
                  } else if (typeof value === 'string') {
                    methods.setValue('voice', value);
                  } else if (value) {
                    methods.setValue('voice', value.value);
                    setSelectedVoiceOption(value);
                  } else {
                    methods.setValue('voice', '');
                  }
                }}
                renderOption={(props, option, { selected }) => (
                  <li {...props}>
                    <Box
                      sx={{
                        mr: 1,
                        width: 32,
                        height: 32,
                        overflow: 'hidden',
                        borderRadius: '50%',
                        position: 'relative',
                      }}
                    >
                      <Avatar alt={option.name} src={option.avatar} sx={{ width: 1, height: 1 }} />
                      <Stack
                        alignItems="center"
                        justifyContent="center"
                        sx={{
                          top: 0,
                          left: 0,
                          width: 1,
                          height: 1,
                          opacity: 0,
                          position: 'absolute',
                          bgcolor: (theme) => alpha(theme.palette.grey[700], 0.6),
                          transition: (theme) =>
                            theme.transitions.create(['opacity'], {
                              easing: theme.transitions.easing.easeInOut,
                              duration: theme.transitions.duration.shorter,
                            }),
                          ...(selected && {
                            opacity: 1,
                            color: 'primary.main',
                          }),
                        }}
                      >
                        <Iconify icon="eva:checkmark-fill" />
                      </Stack>
                    </Box>
                    {option.name}
                  </li>
                )}
              />
            </Stack>
            <Stack sx={{ mt: 1, mb: 5 }} spacing={1}>
              <Typography variant="subtitle2">Speaking style</Typography>
              <FormAutocomplete
                size="small"
                name="style"
                placeholder="Select style"
                freeSolo
                autoSelect
                options={styleOptions}
                getOptionLabel={(option) => option}
                isOptionEqualToValue={(option, value) => option === value}
                renderOption={(props, option) => (
                  <li {...props} key={option}>
                    {option}
                  </li>
                )}
              />
            </Stack>
          </Box>
        </FormProvider>

        <Divider />

        <DialogActions sx={{ p: 3 }}>
          <LoadingButton
            loading={isSubmitting}
            disabled={false}
            type="submit"
            variant="contained"
            onClick={onSubmit}
          >
            Confirm
          </LoadingButton>
          <Button variant="outlined" color="inherit" onClick={onClose}>
            Cancel
          </Button>
        </DialogActions>
      </Stack>
    </Dialog>
  );
}
