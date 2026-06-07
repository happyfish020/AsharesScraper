cn_market full mirror BAT

Usage:
1. Edit full_mirror_cn_market.bat
   - set MYSQL_USER=root
   - set MYSQL_PWD=your password
2. Run from cmd:
   full_mirror_cn_market.bat

This script does not create a .sql dump file. It streams mysqldump directly into mysql.
It drops and recreates cn_market, then imports cn_market_red into cn_market.
Any command failure exits with code 1.
