import pandas as pd
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits

def save_fits(name):
    data = eval(name)
    dat = np.array(data)
    pt = './fits/'    
    if not os.path.exists(pt):
        os.mkdir(pt)
    resu = pt+name+'.csv'
    if os.path.exists(resu):
        os.remove(resu)
    data[data.columns[2]] = data[data.columns[2]].astype('bytes')
    data.to_csv(resu)


df_auto = pd.read_csv('resu.csv')
aa = df_auto.columns
bb = [i.replace('-','_').replace('ra_1','ra').replace('dec_1','dec') for i in aa]
df_auto.columns = bb

yu = 0.6
##merger
merger = df_auto.query('merging_minor_disturbance_fraction > %f '
                                '| merging_major_disturbance_fraction > %f '
                                '| merging_merger_fraction > %f '
                                % (yu, yu, yu))[["ra","dec","iauname"]]

nomerge = df_auto.query('disk_edge_on_no_fraction > %f '
                        '& merging_none_fraction > %f' % (0.6, 0.8))[["ra","dec","iauname"]]


save_fits('merger')
save_fits('nomerge')

smoothRounded = df_auto.query('smooth_or_featured_smooth_fraction >  %f '
                        '& how_rounded_round_fraction > %f'
                                                    % (yu, yu))[["ra","dec","iauname"]]

smoothInBetween = df_auto.query('smooth_or_featured_smooth_fraction >  %f '
                        '& how_rounded_in_between_fraction > %f'
                                                    % (yu, yu))[["ra","dec","iauname"]]

smoothCigarShaped = df_auto.query('smooth_or_featured_smooth_fraction > %f '
                                  '& how_rounded_cigar_shaped_fraction > %f'
                                                              % (yu, yu))[["ra","dec","iauname"]]
save_fits('smoothRounded')
save_fits('smoothInBetween')
save_fits('smoothCigarShaped')
print(smoothRounded.shape[0]+smoothInBetween.shape[0]+smoothCigarShaped.shape[0])

## edge on
edgeOnz = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                            '& disk_edge_on_yes_fraction > %f'
                                                        % (yu, yu))[["ra","dec","iauname"]]
edgeOnno = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                            '& disk_edge_on_no_fraction > %f'
                                                        % (yu, yu))[["ra","dec","iauname"]]
save_fits('edgeOnz')
save_fits('edgeOnno')
### Bar
BarNo = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '& bar_no_fraction > %f '
                                % (yu, yu, yu))[["ra","dec","iauname"]]

BarWeak = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '& bar_weak_fraction > %f '
                                % (yu, yu, yu))[["ra","dec","iauname"]]

BarStrong = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '&bar_strong_fraction > %f '
                                % (yu, yu, yu))[["ra","dec","iauname"]]
save_fits('BarNo')
# save_fits('BarWeak')
save_fits('BarStrong')
print(BarNo.shape[0]+BarWeak.shape[0]+BarStrong.shape[0])

##arm Tin

arms_tight = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '&has_spiral_arms_yes_fraction > %f '
                                '&spiral_winding_tight_fraction > %f '
                                % (yu, yu, yu, yu))[["ra","dec","iauname"]]

arms_medium = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '&has_spiral_arms_yes_fraction > %f '
                                '&spiral_winding_medium_fraction > %f '
                                % (yu, yu, yu, yu))[["ra","dec","iauname"]]

arms_loose = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '&has_spiral_arms_yes_fraction > %f '
                                '&spiral_winding_loose_fraction > %f '
                                % (yu, yu, yu, yu))[["ra","dec","iauname"]]
save_fits('arms_tight')
save_fits('arms_medium')
save_fits('arms_loose')
print(arms_tight.shape[0]+arms_medium.shape[0]+arms_loose.shape[0])
### bulge shape
# edgeOn_boxy = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
#                             '& disk_edge_on_yes_fraction > %f'
#                             '& edge_on_bulge_boxy_fraction > %f'
#                             % (yu, yu, yu))[["ra","dec","iauname"]]

# edgeOn_none = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
#                             '& disk_edge_on_yes_fraction > %f'
#                             '& edge_on_bulge_none_fraction > %f'
#                             % (yu, yu, yu))[["ra","dec","iauname"]]

# edgeOn_round = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
#                             '& disk_edge_on_yes_fraction > %f'
#                             '& edge_on_bulge_rounded_fraction > %f'
#                             % (yu, yu, yu))[["ra","dec","iauname"]]
# save_fits('edgeOn_boxy')
# save_fits('edgeOn_none')
# save_fits('edgeOn_round')
# print(edgeOn_boxy.shape[0]+edgeOn_none.shape[0]+edgeOn_round.shape[0])

# ### have arm
armsz = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '&has_spiral_arms_yes_fraction > %f '
                                % (yu, yu, yu))[["ra","dec","iauname"]]

armsno = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '&has_spiral_arms_no_fraction > %f '
                                % (yu, yu, yu))[["ra","dec","iauname"]]
save_fits('armsz')
save_fits('armsno')
print(armsz.shape[0]+armsno.shape[0])

# ### Bulge
BulgeNo = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '& bulge_size_none_fraction > %f '
                                % (yu, yu, yu))[["ra","dec","iauname"]]

Bulgesmall = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '& bulge_size_small_fraction > %f '
                                % (yu, yu, yu))[["ra","dec","iauname"]]

Bulgemoderate = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '& bulge_size_moderate_fraction > %f '
                                % (yu, yu, yu))[["ra","dec","iauname"]]
save_fits('BulgeNo')
save_fits('Bulgesmall')
save_fits('Bulgemoderate')
print(BulgeNo.shape[0]+Bulgesmall.shape[0]+Bulgemoderate.shape[0])