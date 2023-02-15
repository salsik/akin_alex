#!/bin/bash
for i in {0..10..2}
do
  echo "run test $i times"
done


#for (( c=0.1; c<=0.7; c+=0.2 ))

#for i in $(seq 0.3 -0.2 0.1)

for i in {0.0001,0.0002,0.0003}
do
    echo "run main $i times"
    #value = 4.0
    #d=`expr $i*$value`.
    #val = python -c "print(20+5/2.0)"
    d=$(python -c "print($i*4.0)")
    # same desc add gen lr
    python main.py --train --g_lr $i --d_lr $i
done

### last command
###  python main.py --train --g_lr 0.0002 --d_lr 0.0002 --restore_model