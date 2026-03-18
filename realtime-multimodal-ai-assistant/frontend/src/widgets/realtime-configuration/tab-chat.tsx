import * as Yup from 'yup';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import { useEffect, forwardRef, useImperativeHandle } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';

import { useBoolean } from 'src/hooks/boolean';
// import { getStorage, AOAI_CREDENTIAL_KEY } from 'src/hooks/local-storage';

// import Iconify from 'src/widgets/iconify';
import FormProvider, {
  // FormCheckbox,
  FormTextField,
  FormRadioGroup,
  FormAutocomplete,
} from 'src/widgets/tailored-form';

import { IRtConfiguration } from 'src/types/chat';
// import { IAoaiResourceItem } from 'src/types/azure-resource';

// ----------------------------------------------------------------------

// Voice options for GPT Realtime
const gptRealtimeVoices = [
  'Alloy',
  'Ash',
  'Ballad',
  'Cedar',
  'Coral',
  'Echo',
  'Marin',
  'Sage',
  'Shimmer',
  'Verse',
];

// Voice options for Azure Speech Voice Live
const azureVoiceLiveVoices = [
  'ko-KR-JiMinNeural',
  'ko-KR-SeoHyeonNeural',
  'en-US-Aria:DragonHDLatestNeural',
  'zh-CN-XiaoxiaoNeural',
  'zh-CN-Xiaochen:DragonHDLatestNeural',
  'zh-CN-Xiaochen:DragonHDFlashLatestNeural',
  'zh-CN-Yunfan:DragonHDLatestNeural',
  'Xiaoxiao:DragonHDFlashLatestNeural',
  'zh-TW-HsiaoChenNeural',
  'en-US-JennyMultilingualNeural',
];

const voiceHostOptions = [
  { value: 'gpt-realtime', label: 'GPT Realtime' },
  { value: 'voice-live', label: 'Azure Speech Voice Live' },
];

// ----------------------------------------------------------------------

type Props = {
  onClose: VoidFunction;
  configs: IRtConfiguration;
  onUpdate: (config: IRtConfiguration) => void;
  modes: string[];
  selectedMode: string;
  onChangeMode: (newMode: string) => void;
};

const OpenChatTab = forwardRef(({ onClose, configs, onUpdate, selectedMode }: Props, ref) => {
  // const aoaiCredentials: IAoaiResourceItem[] = getStorage(AOAI_CREDENTIAL_KEY);

  // const aoaiResourceNames = aoaiCredentials ? aoaiCredentials.map((item) => item.resourceName) : [];

  const aoaiConfig = useBoolean(true);

  const AoaiSchema = Yup.object().shape({
    'rt-Deployment': Yup.string().required('AOAI deployment is required'),
  });

  const defaultValues: IRtConfiguration = configs;

  const aoaiParamList = Object.keys(defaultValues);

  const methods = useForm({
    resolver: yupResolver(AoaiSchema),
    defaultValues,
  });

  const { handleSubmit, watch, setValue } = methods;

  // Watch the voice host selection to determine which voice options to show
  const selectedVoiceHost = watch('rt-Voice host' as any);

  // Get appropriate voice choices based on selected voice host
  const getVoiceChoices = () => {
    if (selectedVoiceHost === 'voice-live') {
      return azureVoiceLiveVoices;
    }
    return gptRealtimeVoices; // Default to GPT Realtime voices
  };

  // Auto-update voice choice when host changes
  useEffect(() => {
    if (selectedVoiceHost) {
      // Get appropriate voice choices based on selected voice host
      const currentVoiceChoices =
        selectedVoiceHost === 'voice-live' ? azureVoiceLiveVoices : gptRealtimeVoices;

      const currentVoice = watch('rt-Voice choice' as any);

      // Check if current voice is valid for the selected host
      if (!currentVoiceChoices.includes(currentVoice)) {
        // Set to the first voice in the list for the new host
        setValue('rt-Voice choice' as any, currentVoiceChoices[0]);
      }
    }
  }, [selectedVoiceHost, setValue, watch]);

  const onSubmit = handleSubmit(async (data) => {
    try {
      onUpdate(data as IRtConfiguration);
      onClose();
    } catch (error) {
      console.error(error);
    }
  });

  useImperativeHandle(ref, () => ({
    submit: () => onSubmit(),
  }));

  return (
    <FormProvider methods={methods} onSubmit={onSubmit}>
      <Box sx={{ m: 1.25, mt: -0.5, mb: 2 }}>
        {aoaiConfig.value && (
          <>
            {/* {modes.length > 0 && (
                <Box
                  sx={{
                    mx: 4.5,
                    mt: 1,
                    mb: 3,
                    gap: 3,
                    display: 'grid',
                    gridTemplateColumns: { xs: 'repeat(1, 1fr)', md: 'repeat(3, 1fr)' },
                  }}
                >
                  {renderModes}
                </Box>
              )}

              {modes.length > 0 && (
                <Divider sx={{ mx: 4.5, my: 3 }} style={{ borderStyle: 'dashed' }} />
              )} */}

            <Stack sx={{ mx: 2, mt: 2, mb: 3 }} spacing={3}>
              <FormTextField
                multiline
                fullWidth
                size="small"
                maxRows={20}
                name="rt-System message"
                label="System message"
              />
              <FormRadioGroup
                name="rt-Voice host"
                label="Voice Host"
                options={voiceHostOptions}
                row
                spacing={3}
              />
              <Box
                sx={{
                  display: 'grid',
                  gap: 3,
                  gridTemplateColumns: {
                    xs: 'repeat(1, 1fr)',
                    sm: 'repeat(2, 1fr)',
                    md: 'repeat(3, 1fr)',
                    lg: 'repeat(3, 1fr)',
                  },
                }}
              >
                {aoaiParamList
                  .filter((param) => param.startsWith(selectedMode))
                  .map((item) => {
                    if (item.includes('System message')) return null;
                    if (item.includes('Voice host')) return null;
                    if (item.includes('Deployment')) {
                      return null;
                      // return (
                      //   <FormAutocomplete
                      //     key={item}
                      //     size="small"
                      //     name={item}
                      //     label="Deployment"
                      //     options={aoaiResourceNames.map((name) => name)}
                      //     getOptionLabel={(option) => option}
                      //     renderOption={(props, option) => {
                      //       const { resourceName, primary } = aoaiCredentials.filter(
                      //         (_item: IAoaiResourceItem) => _item.resourceName === option
                      //       )[0];

                      //       if (!resourceName) return null;

                      //       return (
                      //         <li {...props} key={`${selectedMode}-${resourceName}`}>
                      //           {primary && (
                      //             <Iconify
                      //               key={`${selectedMode}-${resourceName}`}
                      //               icon="eva:star-fill"
                      //               width={16}
                      //               sx={{ mr: 1 }}
                      //               color="gray"
                      //             />
                      //           )}
                      //           {resourceName}
                      //         </li>
                      //       );
                      //     }}
                      //   />
                      // );
                    }
                    if (item.includes('Voice choice')) {
                      const currentVoiceChoices = getVoiceChoices();
                      return (
                        <FormAutocomplete
                          key={item}
                          size="small"
                          name={item}
                          label={item.replace(`${selectedMode}-`, '')}
                          options={currentVoiceChoices}
                          getOptionLabel={(option) => option as string}
                          renderOption={(props, option) => {
                            const mode = currentVoiceChoices.filter(
                              (_item: string) => _item === option
                            )[0];

                            if (!mode) return null;

                            return (
                              <li {...props} key={`${selectedMode}-${mode}`}>
                                {mode}
                              </li>
                            );
                          }}
                        />
                      );
                    }
                    if (item.includes('Disable audio')) {
                      return null;
                      // return (
                      //   <FormCheckbox
                      //     key={item}
                      //     name={item}
                      //     label={item.replace(`${selectedMode}-`, '')}
                      //   />
                      // );
                    }
                    // Handle VAD parameters with specific input types
                    if (item.includes('VAD Threshold')) {
                      return (
                        <FormTextField
                          key={item}
                          name={item}
                          label="VAD Threshold"
                          type="number"
                          inputProps={{ min: 0, max: 1, step: 0.1 }}
                          fullWidth
                          size="small"
                        />
                      );
                    }
                    if (item.includes('VAD Prefix Padding')) {
                      return (
                        <FormTextField
                          key={item}
                          name={item}
                          label="VAD Prefix Padding (ms)"
                          type="number"
                          inputProps={{ min: 0, step: 50 }}
                          fullWidth
                          size="small"
                        />
                      );
                    }
                    if (item.includes('VAD Silence Duration')) {
                      return (
                        <FormTextField
                          key={item}
                          name={item}
                          label="VAD Silence Duration (ms)"
                          type="number"
                          inputProps={{ min: 0, step: 50 }}
                          fullWidth
                          size="small"
                        />
                      );
                    }
                    return (
                      <FormTextField
                        key={item}
                        name={item}
                        label={item.replace(`${selectedMode}-`, '')}
                        fullWidth
                        size="small"
                      />
                    );
                  })}
              </Box>
            </Stack>
          </>
        )}
        {!aoaiConfig.value && <Divider sx={{ mx: 2 }} />}
      </Box>
    </FormProvider>
  );
});

export default OpenChatTab;
