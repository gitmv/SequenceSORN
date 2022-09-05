import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter([1,1,1], [13.000776835064634, 13.029636188642442, 13.022920627965508], label="KWTA original 5s")

ax.scatter([2, 2, 2], [12.754662404087265, 12.817792188752923, 12.732600130352964], label="ip_new 0.0005")

ax.scatter([3, 3], [12.892193503466691, 12.86721532295437], label="ip new distance sensitive 0.005")

ax.scatter([4, 4], [12.489312640762432, 12.214597861731285], label="ip new distance sensitive 0.0005")

ax.scatter([5], [12.985714692569681], label="WTA without exhaustion_same_mean")

ax.scatter([6, 6], [12.68922527145223, 12.802042203501678], label="old new copy")

ax.scatter([7, 7], [12.245558676465484, 12.102597375069212], label="no ip")

ax.scatter([8, 8], [12.958006754168053, 13.047419452045556], label="original")

ax.scatter([9, 9], [12.92479970467108, 13.033106796578156], label="apply")

ax.scatter([10, 10], [13.03742348525765, 12.907998343528032], label="clone")


ax.scatter([11, 11], [12.979358515840222, 12.905293665893765], label="0")

ax.scatter([12, 12], [12.926199479205495, 12.91606400650184], label="1")

ax.scatter([13, 13], [12.943177407546216, 12.90590296636414], label="10")

ax.scatter([14, 14], [12.947341543306258, 12.914961087420766], label="100")

ax.scatter([15, 15, 15, 15], [12.965013380810458, 12.69800223655885, 12.793889971469767, 12.802191650335766], label="500")

ax.scatter([16, 16], [12.70144061851424, 12.616065006082387], label="1000")


ax.scatter([17, 17], [12.885213599158435, 12.894784260046727], label="100 same inc dec")

ax.scatter([18, 18], [12.658041356003638, 12.625834620844984], label="10 same inc dec")



ax.scatter([19, 19, 19, 19], [12.978734170749451, 12.966011782754093, 13.05689410984935, 13.023540711883388], label="original new imp")


ax.scatter([20, 20, 20, 20], [12.196012608799176, 12.335563191833518, 12.038681251331973, 12.32963666594571], label="same inc dec")

ax.scatter([21, 21, 21, 21], [12.82092431416039, 12.987626009610286, 12.912029966484681, 12.968824128847217], label="different inc dec")


ax.scatter([22, 22], [12.8633901574521, 12.946070131955906], label="ws 100")

ax.scatter([23], [12.923045584302272], label="refractory 100 window")

ax.scatter([24,24,24], [12.816046866572272, 12.968011349547762, 12.863997382825328], label="sw 100 similar speed")









ax.scatter([30,30], [11.779579043842634, 11.318473824597085], label="no comp")

ax.scatter([31,31,31,31,31,31], [12.551132423969342, 12.632227156436898, 12.588603001888481, 12.609818073277154, 12.916702867145627, 12.627037176765672], label="normal")

ax.scatter([32, 32, 32], [12.457566490292777, 12.838506067688893, 12.64065150139893], label="WTA L refrac")

ax.scatter([33, 33, 33, 33], [12.77354772532342, 12.469788405399257, 12.634061607070793, 12.38601902029676], label="0.07 0")

ax.scatter([34, 34, 34, 34], [12.378212909476613, 12.575567043527485, 12.54018549122917, 12.439473113953907], label="0.07 10")

ax.scatter([35, 35, 35, 35], [12.4530984718227, 12.429995762440116, 12.365774368540274, 12.284181468341306], label="0.07 100")





ax.scatter([37, 37, 37, 37], [12.623231347099555, 12.681123392201656, 12.598860409559185, 12.560312830024333], label="0.007 10")

ax.scatter([38, 38, 38, 38], [12.592732167372795, 12.642185216821481, 12.61166843582949, 12.537894685400598], label="0.007 100")



ax.scatter([40, 40, 40, 40], [12.71174026049343, 12.671323877153245, 12.752188725796023, 12.608442927707364], label="0.01 10")

ax.scatter([41, 41, 41, 41], [12.627818877935567, 12.634753978203568, 12.772441074304032, 12.6006655458463], label="0.01 100")


ax.scatter([43, 43, 43, 43], [12.61178038341546, 12.600449645433264, 12.724998565595001, 12.376204051654955], label="0.02 10")

ax.scatter([44, 44, 44, 44], [12.548492893213554, 12.90382984955956, 12.616780587010796, 12.664511524880247], label="0.03 10")

ax.legend()

plt.show()