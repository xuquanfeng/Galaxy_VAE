import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits

def save_fits(name):
    data = eval(name)
    dat = np.array(data)
    pt = '../all/'
    resu = pt+name+'.fits'
    patt = '/data/GZ_Decals/nomerge/180.83411228315592_13.525592564981356_0.fits'
    dds = fits.open(patt)
    if os.path.exists(resu):
        os.remove(resu)
    data[data.columns[2]] = data[data.columns[2]].astype('bytes')
    grey=Table(dat,names=data.columns, dtype=data.dtypes)
    grey = fits.BinTableHDU(grey)
    greyHDU=fits.HDUList([dds[0],grey])
    greyHDU.writeto(resu)

pp = '/data/xuquanfeng/decals_2022/gz_decals_auto_posteriors.parquet'##xingbiao: galaxy table

df_auto = pq.read_table(pp).to_pandas()
aa = df_auto.columns
bb = [i.replace('-','_') for i in aa]
df_auto.columns = bb
merger = df_auto.query('merging_minor_disturbance_fraction > %f '
                               '| merging_major_disturbance_fraction > %f '
                               '| merging_merger_fraction > %f '
                               % (0.6, 0.6, 0.6))[["ra","dec","iauname"]]
save_fits('merger')

# smoothRounded = df_auto.query('smooth_or_featured_smooth_fraction >  %f '
#                         '& how_rounded_round_fraction > %f' % (0.7, 0.8))[["ra","dec","iauname"]]
# save_fits('smoothRounded')

# smoothInBetween = df_auto.query('smooth_or_featured_smooth_fraction >  %f '
#                         '& how_rounded_in_between_fraction > %f' % (0.7, 0.85))[["ra","dec","iauname"]]
# save_fits('smoothInBetween')

# smoothCigarShaped = df_auto.query('smooth_or_featured_smooth_fraction > %f '
#                                   '& how_rounded_cigar_shaped_fraction > %f' % (0.5, 0.6))[["ra","dec","iauname"]]
# save_fits('smoothCigarShaped')

# edgeOn = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
#                             '& disk_edge_on_yes_fraction > %f'
#                             % (0.5, 0.7))[["ra","dec","iauname"]]
# save_fits('edgeOn')

# diskNoBar = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
#                                 '& disk_edge_on_no_fraction > %f '
#                                 '& bar_no_fraction > %f '
#                                 % (0.5, 0.5, 0.8))[["ra","dec","iauname"]]
# save_fits('diskNoBar')

# diskWeakBar = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
#                                 '& disk_edge_on_no_fraction > %f '
#                                 '& bar_weak_fraction > %f '
#                                 % (0.5, 0.5, 0.5))[["ra","dec","iauname"]]
# save_fits('diskWeakBar')

# diskStrongBar = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
#                                 '& disk_edge_on_no_fraction > %f '
#                                 '&bar_strong_fraction > %f '
#                                 % (0.5, 0.5, 0.5))[["ra","dec","iauname"]]
# save_fits('diskStrongBar')
