hatch run python scripts/launcher.py --T 4 --TM 8 --phi $1 --nsteps 48 --nepochs 25 --sigma 0.01 --schema 1 --nsims $2 --save True --normalize True --device gpu --batch_size 4 --device_num $3 --decay 0.995
hatch run python scripts/launcher.py --T 4 --TM 8 --phi $1 --nsteps 48 --nepochs 25 --sigma 0.01 --schema 1 --nsims $2 --save True --normalize True --device gpu --batch_size 8 --device_num $3 --decay 0.995
hatch run python scripts/launcher.py --T 4 --TM 8 --phi $1 --nsteps 48 --nepochs 25 --sigma 0.01 --schema 1 --nsims $2 --save True --normalize True --device gpu --batch_size 16 --device_num $3 --decay 0.995
hatch run python scripts/launcher.py --T 4 --TM 8 --phi $1 --nsteps 48 --nepochs 25 --sigma 0.01 --schema 1 --nsims $2 --save True --normalize True --device gpu --batch_size 32 --device_num $3 --decay 0.995
hatch run python scripts/launcher.py --T 4 --TM 8 --phi $1 --nsteps 48 --nepochs 25 --sigma 0.01 --schema 1 --nsims $2 --save True --normalize True --device gpu --batch_size 64 --device_num $3 --decay 0.995
hatch run python scripts/launcher.py --T 4 --TM 8 --phi $1 --nsteps 48 --nepochs 25 --sigma 0.01 --schema 1 --nsims $2 --save True --normalize True --device gpu --batch_size 128 --device_num $3 --decay 0.995
hatch run python scripts/launcher.py --T 4 --TM 8 --phi $1 --nsteps 48 --nepochs 25 --sigma 0.01 --schema 1 --nsims $2 --save True --normalize True --device gpu --batch_size 256 --device_num $3 --decay 0.995