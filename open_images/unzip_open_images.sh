echo "Unzipping test data"
unzip -q test.zip
echo "Unzipping test challenge data"
unzip -q test_challenge.zip

echo "Unzipping validation data"
unzip -q validation.zip

echo "Unzipping train data"
unzip -q train_00.zip
unzip -q train_01.zip
unzip -q train_02.zip
unzip -q train_03.zip
unzip -q train_04.zip
unzip -q train_05.zip
unzip -q train_06.zip
unzip -q train_07.zip
unzip -q train_08.zip

mkdir zips
mv *.zip zips/