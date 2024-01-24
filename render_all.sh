for file in `ls $1/*.json`
do
  echo $file
  python -m snake render $file > /dev/null 2>&1
done
