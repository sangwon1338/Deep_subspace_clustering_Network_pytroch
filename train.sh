#! /bin/bash
:<<'END'
python Pre_train_Coil20.py --epoch 100

python DSC_Coil20.py --epoch 100 --count 2 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 2 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 2 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 2 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 200

python DSC_Coil20.py --epoch 100 --count 3 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 3 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 3 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 3 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 300

python DSC_Coil20.py --epoch 100 --count 4 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 4 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 4 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 4 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 400

python DSC_Coil20.py --epoch 100 --count 5 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 5 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 5 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 5 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 500

python DSC_Coil20.py --epoch 100 --count 6 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 6 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 6 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 6 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 600

python DSC_Coil20.py --epoch 100 --count 7 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 7 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 7 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 7 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 700

python DSC_Coil20.py --epoch 100 --count 8 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 8 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 8 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 8 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 800

python DSC_Coil20.py --epoch 100 --count 9 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 9 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 9 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 9 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 900

python DSC_Coil20.py --epoch 100 --count 10 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 10 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 10 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 10 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 1000

python DSC_Coil20.py --epoch 100 --count 11 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 11 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 11 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 11 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 1100

python DSC_Coil20.py --epoch 100 --count 12 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 12 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 12 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 12 --name1 H --name2 I

python Pre_train_Coil20.py --epoch 1200

python DSC_Coil20.py --epoch 100 --count 13 --name1 B --name2 C
python DSC_Coil20.py --epoch 100 --count 13 --name1 D --name2 E
python DSC_Coil20.py --epoch 100 --count 13 --name1 F --name2 G
python DSC_Coil20.py --epoch 100 --count 13 --name1 H --name2 I


python Pre_train_Coil20.py --epoch 100
python Pre_train_Coil20.py --epoch 200
python Pre_train_Coil20.py --epoch 300
python Pre_train_Coil20.py --epoch 400
python Pre_train_Coil20.py --epoch 500
python Pre_train_Coil20.py --epoch 600
python Pre_train_Coil20.py --epoch 700
python Pre_train_Coil20.py --epoch 800
python Pre_train_Coil20.py --epoch 900
python Pre_train_Coil20.py --epoch 1000
python Pre_train_Coil20.py --epoch 1100
python Pre_train_Coil20.py --epoch 1200
END

python DSC_Coil20.py --epoch 50 --count 2 --pre 100 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 3 --pre 200 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 4 --pre 300 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 5 --pre 400 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 6 --pre 500 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 7 --pre 600 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 8 --pre 700 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 9 --pre 800 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 10 --pre 900 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 11 --pre 1000 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 12 --pre 1100 --name1 L --name2 M
python DSC_Coil20.py --epoch 50 --count 13 --pre 1200 --name1 L --name2 M
