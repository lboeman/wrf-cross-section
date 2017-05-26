# wrf-cross-section

## Usage
  * Environment Variables:
    * MYSQL_CRED: The path to a file containing mysql login credentials in json format. Defaults to '~/.mysql'.

      example:
      ```
      {
        "user": "user",
        "passwd": "password",
        "host": "127.0.0.1",
        "port": 3306
      }
      ```
    * WRF_DATA_DIRECTORY: The path to the directory to find WRF data in. Defaults to '/a4/uaren'.
  * Serve cross sections with bokeh using:
