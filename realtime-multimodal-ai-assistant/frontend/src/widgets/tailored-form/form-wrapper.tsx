import { FormProvider, UseFormReturn } from 'react-hook-form';

// ----------------------------------------------------------------------

interface FormWrapperProps {
  children: React.ReactNode;
  methods: UseFormReturn<any>;
  onSubmit?: () => void;
}

const FormWrapper = ({ children, methods, onSubmit }: FormWrapperProps) => (
  <FormProvider {...methods}>
    <form onSubmit={onSubmit}>{children}</form>
  </FormProvider>
);

export default FormWrapper;
