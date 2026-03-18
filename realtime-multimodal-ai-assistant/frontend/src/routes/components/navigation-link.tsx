import { forwardRef } from 'react';
import { Link, LinkProps } from 'react-router-dom';

// ----------------------------------------------------------------------

interface NavigationLinkProps extends Omit<LinkProps, 'to'> {
  path: string;
  component?: React.ElementType;
}

const NavigationLink = forwardRef<any, NavigationLinkProps>(({ path, component = Link, ...other }, ref) => {
  const Component = component;
  return <Component ref={ref} to={path} {...other} />;
});

export default NavigationLink;