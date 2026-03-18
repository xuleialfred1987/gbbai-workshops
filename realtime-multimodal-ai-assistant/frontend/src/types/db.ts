export type MySqlConnProps = {
  connectionName: string;
  host: string;
  port: string;
  userName: string;
  password: string;
  tables?: string[];
};
