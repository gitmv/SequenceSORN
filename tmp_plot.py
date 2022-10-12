import matplotlib.pyplot as plt
import numpy as np
import random



plt.plot([0.25],[0.025],'.')
plt.show()




'''
N = 10000
xx = np.linspace(2, 4, N)
yy = np.zeros(N)
for jj in range(N):
    a = xx[jj]
    y0 = random.random()
    for ii in range(1, 100):
        y0 = a * y0 * (1 - y0)
    yy[jj] = y0
plt.plot(xx,yy,'.')
plt.show()
'''













'''
txt = 'xx         e  eee ea eaaeaaaaaaaatatttttttt....... ..            ffffeffofooeoooooxxxxxxx   x       eeeeeeeeaaaeeaaaaaatttttt#tttttsssss.ss s           fmmffeffmoeooeoooooxxxxxxx   #        efffefeeeeeeaaaaaatattatttttt.s..s..s.ss               f mfeeffeeooooooooxxxxxxxxx #           fmemfeea#eeaaatttt####t.tts....####ss       ##    eee###eeeeeaaa#eaaaeaatttttaatttts.....s   #        #mmm memmeeeeeeeaaaaaaaaatatt#ttttts.sss#.s .            f mffooooooxxxxx   x        m ee e efeeeeeaaaaaatattttttttst.....s.s            m  mffmfmmmeeoeeoooooeooaxxxxxxx   x     f e    fffeaoaaaeaaootoaaatttattttt.tss.s...  #            f mmemfffmeeooooeeeaaaeaaaaattttttttttt........ ss            m m    eeeae#eeeaaaaaa#aeatttttt.s#atttssssss.. s            ffffffofoeoooxxxx   #           emffeeeeeeeaaeaaaaaatatttttttt.tts.s...s .               # m eeeeeeaaaaaaaaaaatttttttt.ttt..ss...           mfmfmfooooooxxxxx                 fe eeeemeeeeeaaaa#aaaaatttt#ttttatttssss. .#..        #    ffffffoeoooxxxxx    ####        mm### meeeeeeaeaaaaaaaatattttttttttat.....#.              mfmfmeeeeeeeeeaeaaaaaaaattttttttstsssssssss           ffmffmfooooxxxxxx######        mmfeeeeefeeaeaeaaaaatatattttttt......#...            mmmmmee emeeeeaeaaeaaaaaatttattttttt............          fff oooooooxxxxxxxxx            e emeeeeeeaeaaaaaaaaatttattttttts.sssssssss             mfmffoooooooooxxxxxxx          me eeee#eeeeaaaa##aaattttttt##ttt........ s          f fffooooooooooxxxxxxx x        e    mmmeameeeeaaa##aaaaaaatt#ttaatttt.s.s.sss                ff##f## eeeeaeaeeeaaaaaaaatatttttttt.....s..s              ffffofofooooxxxxx x               eeeeeeaaaaaaaatttt####ttttt.ss...s...             mm#f#eeeeeeeeaeeeaaaattttt####tttatts..sss .           fm ffffoooooxxxxxxxxxx          eeeeeeeeeeeaaaaaaatattttttttttt.....s.             m  mmeememmeeeeeaaa###aaaaattttatatttt.........      s    ff  mefooooooxxxoxxx   x ##       feeee aeeae#aaeaaattt###attttts.s#s##s#.s           fmfm###foeeoooooxxxxxxxxx      #     eefeefeeeaeaaa##aaaattattttttt.....ss               mmmmmmmeeee#####eeeaaa#aaaaaattttt.tttttssssssssss           f f  efeeoooooooxxxx    ##        f  f emeeeeeeeaaaaatt###aatttt..s......            ff#mffeeeeeee#eaaaaaaaataattttttttss.sssss  s      f      fmffeeofooooxxxxxxxx    #          fefeeeeoeaaaaeaaaaaaaattttttt.t.....s   ##             mfffmemeeeeaeeeaaaaaaaeaattttttttttt..s....#..             mmfmeffoeooooxxxxx  #  #             eeeeeeeeeaaaaaaaaaattttttttttstsss... s              fmffeeeeeeeaeeaaaa####aatttttttttttt......s. .           m   fffeeffoooooxxxx   ## #            mfeeeee#aaaaaaaaaattttttttttttttssssss#  .         fffmfomeoeoooxoxxxxx#oxxx            eefmmeeeeeaaaaaaaatttttttatttt.t....t..             mffeffeeffoooooooooxxxxx            mee  ea eaeaaeaaaaattttt###ttt...ss..s  #           f m meeee####eeeaaaaaaaaatatttttttttttsssss  ##.s      m     mmfmmeeeeeeaaa##a#aaaaatata##ttttttt........s.            eeef#offooooxoxxxxx x            fee oaeefeooaoooaooxxxxxxx             m ffmmfeeeeeeeeaaaaaaatttttttttttts.s.sss .               ffmfffefeoooooooxxxxxxxxxx      e     meemeaeaeaaaa##t#aatttttt.s...s....s                fmmmmmeeeeeeeeeaaaaaat#ttattttttttt....ss..s.             mmeeefeeeeeee#eaaaaaaaaatatttttttttts.......  s             fffofo#ooooxxxx     ######       efeeeeeeeeaeaaaaaaattttttttttts.ss.#s#s.s            mm memeeeeeeeaeeaaaaaaaatattttttt.ts..ss.sss.     #      fffffooooooxxxxxxxxxx         fefeeaaaeeeeeaaaaaatatattttttttt.tt......s  .           mmfmmfom##oeoeeooxooxxxxxxxxx     #     eeee####eeaeaaaaaaaaaatttttsttss.tssss.s s #        mmmmmmeefeemeeeeaeeaaaaaaaaaaattttt##s###t.s..s.    #        ffffff foooxxx     #######       fme##eeeeeeeeaaaeaaaaatt##ttttttttt....s#.....      f ##   fmmmmoommeeeeeeeeeeeaaaaaaaaaattttttt#tttts.sssssss         #mm ffffmooooooxo#oeoxoxxxxx            ffeeefeeeeeaeeeaeaaaaaaaaaaatttttt.ttt.....   .          m  mmmfmfemeeoeeeeeeaaaaaaaaaatttatatttttssssssss   s        f m mmeeefeeeeooooooxxxxxxxx            e e e  eeaaaaaaaaaataattttttt...... .               fmmom emfeeeeeoo######eaaaaaataaattttttttttsss.   #####       m##m f  efe#f#feeeeeoeeeoaaooeaaaaaaatattttt.tt.tts.s.....              mmmm  eeeeeeeeeaaaaaaaaaaaaaattttttttttt..ssss..        ##     mmmem#meemeeeeeeeeeaaaaaaaaaaatttt##tttttts.s..#s#         ##    fffofoooooxxx    ########### eeeeeeeeeeeaaaaaaaaaaattttt#tttttttts...s.ss .           ffffmofoooooooxxxx    #  #      emefm eeeeeeaaaaaaaaaaa#a#aattttt..sss.##s.s .           m mmemmeeeeeeeeeeaaaaaaaaaaaattttttttttts.sssss###.            fffffoooxxxxxxxxxxoxxx    #####     eeeee###eeeeeeaaaata####atttttttttt.t.ts.s...s            fmfmmmfmoooooooooooxxxxxxxxxx          eeeeeeaeaeaaaaaaaaatattttttttt.ss...  ###.            mfmfemfemee#eeeeaeoooeaaaaaaaaaatttttttssstsss...  ###          f f fffeefeoeeoooooooxxxxxxx             eeeeeeeeaeeaaaaaa#aaaatttt.t...tt.sssss ##         fmfm fmfmmfoooooooooeooeeoxoexeeaaaaaaaaaaaatttttt.tttt.t..s    ####       fmfm#m#mefeeeeeeeeeeeaeaaaaa'
txt = list(txt)
chars = np.unique(txt)
c_vec = np.zeros(len(chars))

char_dict = {}
for c in np.unique(txt):
    char_dict[c] = 0.0


new_txt = ''

for c in txt:
    char_dict[c] += 2.0

    max = chars[0]
    for k in char_dict:
        if char_dict[k]>0:
            char_dict[k] -= 1.0

        if char_dict[k] > char_dict[max]:
            max = k

    new_txt += max

print(char_dict)

for i in range(10):
    for c in chars:
        new_txt = new_txt.replace(c+c, c)

#for _ in range(3):
#    for i in range(len(txt)-2):
#        txt[i]
#        if txt[i]==txt[i+2]:
#            txt[i+1] = txt[i+2]




print(new_txt)
#print(''.join(txt))




'''


'''
class cl:

    def a(self, x,y):
        print(x,y)

    def b(self, z, w, *args, **kwargs):
        self.a(*args, **kwargs, x='f')
        print(z, w)

    def set_functions(self):

        def abc(self, eee):
            print(eee)

        self.abc = abc

g = cl()
g.set_functions()

g.b(y='y', z='z', w='w')
g.abc('sdfasdf')



x=[]
t=1
for i in range(30):
   x.append(t)
   t=t-t*0.2+(1-t)*0.05

for i in range(60):
   x.append(t)
   t=t+(1-t)*0.05

plt.plot(x)


x=[]
t=0
for i in range(30):
   x.append(1-t)
   t=np.clip(t+0.173,0,1)

for i in range(60):
   x.append(1-t)
   t=np.clip(t-0.173,0,1)

plt.plot(x)


plt.show()
'''

