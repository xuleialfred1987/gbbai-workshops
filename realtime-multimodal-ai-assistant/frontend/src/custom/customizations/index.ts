import merge from 'lodash/merge';

// mui
import { Theme } from '@mui/material/styles';

// project imports
import * as components from './mui-widgets';
import { defaultProps } from './default-props';

// ----------------------------------------------------------------------

export function componentsOverrides(theme: Theme) {
  // Create an array of component customization functions
  const customizationFunctions = [
    defaultProps,
    components.fab,
    components.tabs,
    components.chip,
    components.card,
    components.menu,
    components.list,
    components.table,
    components.paper,
    components.alert,
    components.badge,
    components.radio,
    components.select,
    components.button,
    components.rating,
    components.dialog,
    components.appBar,
    components.avatar,
    components.slider,
    components.drawer,
    components.tooltip,
    components.popover,
    components.stepper,
    components.svgIcon,
    components.switches,
    components.checkbox,
    components.dataGrid,
    components.progress,
    components.skeleton,
    components.timeline,
    components.backdrop,
    components.textField,
    components.accordion,
    components.typography,
    components.pagination,
    components.datePicker,
    components.breadcrumbs,
    components.cssBaseline,
    components.autocomplete,
    components.toggleButton,
    components.loadingButton,
  ];

  // Apply each customization function to the theme and merge results
  return customizationFunctions.reduce((acc, fn) => merge(acc, fn(theme)), {});
}
