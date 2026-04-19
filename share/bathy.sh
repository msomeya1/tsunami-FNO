wget https://kiyuu.bosai.go.jp/GtTM/WholeJapan2020.grd
# gmt grdinfo WholeJapan2020.grd

gmt grdcut WholeJapan2020.grd -R131.2/136.2/29.5/34.5 -GHyuganada_cut.grd=cf
gmt grdfilter Hyuganada_cut.grd -D0 -Fg0.06 -I0.02/0.02 -GHyuganada.grd=cf
# gmt grdinfo Hyuganada.grd