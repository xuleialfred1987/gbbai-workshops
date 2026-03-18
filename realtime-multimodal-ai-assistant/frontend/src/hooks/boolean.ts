import { useReducer, useCallback } from 'react';

// ----------------------------------------------------------------------

type BooleanAction =
  | { type: 'SET_TRUE' }
  | { type: 'SET_FALSE' }
  | { type: 'TOGGLE' }
  | { type: 'SET_VALUE'; payload: boolean };

interface BooleanState {
  value: boolean;
}

interface ReturnType {
  value: boolean;
  onTrue: () => void;
  onFalse: () => void;
  onToggle: () => void;
  setValue: React.Dispatch<React.SetStateAction<boolean>>;
}

export function useBoolean(defaultValue?: boolean): ReturnType {
  // Create reducer for boolean state management
  const booleanReducer = (state: BooleanState, action: BooleanAction): BooleanState => {
    switch (action.type) {
      case 'SET_TRUE':
        return { value: true };
      case 'SET_FALSE':
        return { value: false };
      case 'TOGGLE':
        return { value: !state.value };
      case 'SET_VALUE':
        return { value: action.payload };
      default:
        return state;
    }
  };

  const [state, dispatch] = useReducer(booleanReducer, { value: Boolean(defaultValue) });

  // Create memoized action dispatchers
  const onTrue = useCallback(() => dispatch({ type: 'SET_TRUE' }), []);
  const onFalse = useCallback(() => dispatch({ type: 'SET_FALSE' }), []);
  const onToggle = useCallback(() => dispatch({ type: 'TOGGLE' }), []);

  // Handle React's setState API compatibility
  const setValue = useCallback(
    (value: React.SetStateAction<boolean>) => {
      const newValue = typeof value === 'function' ? value(state.value) : value;

      dispatch({ type: 'SET_VALUE', payload: newValue });
    },
    [state.value]
  );

  return {
    value: state.value,
    onTrue,
    onFalse,
    onToggle,
    setValue,
  };
}
