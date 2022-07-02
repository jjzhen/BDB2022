#%% BDB2022.py

'''

BLUE DEVIL BUDDIES
2022 MATCHING ALGO

Created Thu Jun 24 07:41:26 2021
Bennett David & James Zheng

!git clone https://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@github.com/jjzhen/BDB2021.git
!git add 'BDB2021.py'
!git commit -m 'BDB2021.py'
!git push

'''

#%% PACKAGES

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import datetime as dt
import matplotlib.pyplot as plt

#%% DATA IMPORT
def import_qualtrics_results(fname): 
    tbl = pd.read_csv(fname)
    tbl = tbl.drop_duplicates(subset = ['NetID'], keep = 'last')
    tbl.fillna('', inplace = True)
    
    tbl_P = tbl.loc[tbl['Q11'] == 'Pratt']
    tbl_T = tbl.loc[tbl['Q11'] == 'Trinity']
    tbl_P = tbl_P.reset_index(drop=True)
    tbl_T = tbl_T.reset_index(drop=True)
    return tbl, tbl_P.to_dict('index'), tbl_T.to_dict('index')

tbl_tees, mentees_P, mentees_T = import_qualtrics_results('mentee.csv')
tbl_tors, mentors_P, mentors_T = import_qualtrics_results('mentor.csv')
#%% CLEANING FCN

def clean(data):

    for key in data:
        
        # time zone?
        if key == 'Q9':
            if data['Q9'][0:3] == 'GMT' or data['Q9'][0:3] == 'UTC':
                data['Q9'] = 0
            elif data['Q9'].rsplit('GMT')[-1] == '':
                pass
            else:
                textzone = data['Q9'].rsplit('GMT')[-1]
                numbzone = textzone.rsplit(':')
                data['Q9'] = int(numbzone[0]) + int(numbzone[1])/60
        
        # in touch over summer?
        if key == 'Q33':
            if 'Once' in data['Q33']:
                data['Q33'] = 0
            elif '2' in data['Q33']:
                data['Q33'] = 1
            else:
                data['Q33'] = 2
                
        # in touch over semester?       
        if key == 'Q34':
            if 'Once a month' in data['Q34']:
                data['Q34'] = 0
            elif 'times a month' in data['Q34']:
                data['Q34'] = 1
            elif 'Once a week' in data ['Q34']:
                data['Q34'] = 2
            else:
                data['Q34'] = 3
        
        # what kind of relationship?
        if key == 'Q35':
            if 'Help' in data['Q35']:
                data['Q35'] = 0
            elif 'Give' in data ['Q35']:
                data['Q35'] = 1
            elif 'Catch' in data['Q35']:
                data['Q35'] = 2
            else:
                data['Q35'] = 3
        
        # ideal saturday night?
        if key == 'Q26':
            if 'Reading' in data['Q26']:
                data['Q26'] = 0
            elif 'Watching' in data['Q26']:
                data['Q26'] = 1
            elif 'Playing' in data ['Q26']:
                data['Q26'] = 2
            elif 'Exploring' in data['Q26']:
                data['Q26'] = 3
            elif 'Wine' in data ['Q26']:
                data['Q26'] = 4
            else:
                data['Q26'] = 5
        
        # someone who drinks alcohol?
        if key == 'Q21':
            if 'No' in data['Q21']:
                data['Q21'] = 0
            elif '1-3' in data['Q21']:
                data['Q21'] = 1
            elif '4-6' in data ['Q21']:
                data['Q21'] = 2
            else:
                data['Q21'] = 3
        
        # someone who smokes weed?
        if key == 'Q22':
            if 'No' in data['Q22']:
                data['Q22'] = 0
            elif 'light' in data['Q22']:
                data['Q22'] = 1
            elif 'moderate' in data ['Q22']:
                data['Q22'] = 2
            else:
                data['Q22'] = 3
    
        # level of basketball excitement?
        if key == 'Q30':
            if 'Who' in data['Q30']:
                data['Q30'] = 0
            elif 'might' in data['Q30']:
                data['Q30'] = 1
            elif 'mind' in data ['Q30']:
                data['Q30'] = 2
            else:
                data['Q30'] = 3

        if key == 'Q18': 
            programs_clean = []
            if 'Bass Connections' in data['Q18']:
                programs_clean += ['Bass Connections']
            if 'DukeEngage' in data['Q18']:
                programs_clean += ['DukeEngage']
            if 'FOCUS' in data['Q18']:
                programs_clean += ['FOCUS']
            if 'Study Abroad' in data['Q18']:
                programs_clean += ['Study Abroad']
            data['Q18'] = ", ".join(programs_clean).strip()

#%% COMPARING FCN (~1260 total pts)

def compare(mentee, mentor, prppts, grnpts, ylwpts):
    
    prpscore = 0
    grnscore = 0
    ylwscore = 0
    rawscore = 0
    
    # which identities would you like to have in common w/your mentor?
    #if 'Abil' in mentee['Q44'] and mentee['Q42'] == mentor['Q41']:
        #rawscore += prppts
    if 'Eth' in mentee['Q44'] and mentee['Q37'] == mentor['Q37']:
        rawscore += prppts
    if 'Diet' in mentee['Q44'] and mentee['Q43'] == mentor['Q42']:
        rawscore += prppts
    if 'First' in mentee['Q44'] and mentee['Q39'] == mentor['Q39']:
        rawscore += prppts
    if 'Gend' in mentee['Q44'] and mentee['Q36'] == mentor['Q36']:
        rawscore += prppts
    if 'Limit' in mentee['Q44'] and mentee['Q40'] == mentor['Q52']:
        rawscore += prppts
    if 'Reli' in mentee['Q44'] and mentee['Q41'] == mentor['Q40']:
        rawscore += prppts
    if 'Orien' in mentee['Q44'] and mentee['Q38'] == mentor['Q38']:
        rawscore += prppts
    
    # what kind of relationship are you looking for?
    if abs(mentee['Q35'] - mentor['Q35']) <= 1:
        rawscore += prppts    

    # what is your ideal saturday night?
    if abs(mentee['Q26'] - mentor['Q26']) <= 1:
        rawscore += prppts

    # do you want someone who drinks?
    if abs(mentee['Q21'] - mentor['Q21']) <= 1:
        rawscore += prppts
    
    # do you want someone who smokes weed?
    if abs(mentee['Q22'] - mentor['Q22']) <= 1:
        rawscore += prppts
    prpscore = rawscore
    
    # would you be willing to mentor a transfer student (tor)?
    if 'Yes' in mentor['Q48'] and 'transfer' in mentee['Q3']:
        rawscore += grnpts
    elif 'Yes' in mentor['Q58'] and 'gap-year' in mentee['Q3']: 
        rawscore += grnpts
    #else:
        #if 'first-year' in mentee['Q3']:
            #rawscore += grnpts
    
    # are you in Trinity or Pratt?
    #if mentee['Q11'] == mentor['Q11']:
        #rawscore += grnpts
    
    # what is your intended major/minor/certificate?
    if 'Academic' in mentee['Q44']: 
        rawscore += prppts*len(set(mentee['Q12'].split(',')) & set(mentor['Q12'].split(',')))
        rawscore += grnpts*len(set(mentee['Q13'].split(',')) & set(mentor['Q13'].split(',')))
        rawscore += grnpts*len(set(mentee['Q14'].split(',')) & set(mentor['Q14'].split(',')))
    else: 
        rawscore += grnpts*len(set(mentee['Q12'].split(',')) & set(mentor['Q12'].split(',')))
        rawscore += ylwpts*len(set(mentee['Q13'].split(',')) & set(mentor['Q13'].split(',')))
        rawscore += ylwpts*len(set(mentee['Q14'].split(',')) & set(mentor['Q14'].split(',')))
    
    # what is your time zone?
    if abs(mentee['Q9'] - mentor['Q9']) <= 3:
        rawscore += grnpts
    
    # how often would you like to be in touch over the summer?
    if abs(mentee['Q33'] - mentor['Q33']) <= 1:
        rawscore += grnpts

    # how often would you like to be in touch during the semester?
    if abs(mentee['Q34'] - mentor['Q34']) <= 1:
        rawscore += grnpts

    # what are you most looking forward to about coming to college?
    if mentee['Q19'] == mentor['Q50']:
        rawscore += grnpts
   
    # what are you most nervous about coming to college?
    if mentee['Q20'] == mentor['Q19']:
        rawscore += grnpts
 
    # what are you interested in at Duke?
    for interest in mentee['Q15'].split(','):
        if interest in mentor['Q15']:
            rawscore += grnpts
    grnscore = rawscore - prpscore
    
    #Quad-Ex
    linkages = {'Craven': ['Bassett', 'Pegram'], 
                'Crowell': ['Giles', 'Wilson'], 
                'Edens': ['Trinity', 'Bell Tower'], 
                'Few': ['Gilbert-Addoms', 'Southgate'], 
                'Keohane': ['Blackwell', 'Randolph'], 
                'Kilgo': ['Alspaugh', 'Brown'], 
                'Wannamaker': ['East House', 'West House']}
    
    if 'Unsure' not in mentee['Q56'] and mentor['Q57'] in linkages.keys(): 
        mentor_quad = mentor['Q56'] if mentor['Q56'] != '' else mentor['Q57']
        if mentee['Q56'] in linkages.get(mentor_quad): 
            rawscore += grnpts
    
    if 'Extracurricular' in mentee['Q44']: 
        interestpts = grnpts
    else: 
        interestpts = ylwpts
        
    for interest in mentee['Q16'].split(','):
        if interest in mentor['Q17']:
            rawscore += interestpts
    for interest in mentee['Q17'].split(','):
        if interest in mentor['Q16']:
            rawscore += interestpts
    for interest in mentee['Q18'].split(','):
        if interest in mentor['Q18']:
            rawscore += interestpts
    for interest in mentee['Q23'].split(','):
        if interest in mentor['Q23']:
            rawscore += interestpts
    
    # how crazy are you about Duke Basketball?
    if abs(mentee['Q30'] - mentor['Q30']) <= 1:
        rawscore += ylwpts
    
    # have you had siblings attend Duke within the past 5 years?
    if mentee['Q32'] == 'Yes' and mentor['Q33'] + mentor['Q34'] <= 2:
        rawscore += ylwpts
    ylwscore = rawscore - prpscore - grnscore
    
    return [rawscore, prpscore, grnscore, ylwscore]

#%% HUNGARIAN FCN

def hunger(scores, filtered_input = False, tee_dict = {}, tor_dict = {}):
    
    invscores = totpts - scores
    matches = []
    tees, tors = linear_sum_assignment(invscores)
    if filtered_input and len(tee_dict.keys()) != len(tees):
        #print(len(scores), tees, tors)
        print("Error: length mismatch {} vs {}".format(len(tee_dict.keys()), len(tees)))
        print("There are unmatched mentees, which makes additional rounds of pairing impossible.")
        
    for x in range(len(tees)):
        
        teeindex = tees[x]
        torindex = tors[x]
        
        if not filtered_input: 
            teei = teeindex
            tori = torindex 
        else: 
            teei = tee_dict[teeindex]
            tori = tor_dict[torindex]
        
        mentee = mentees.get(teei) # lowest avg matching to this mentee
        mentor = mentors.get(tori) # best mentor in mentee's row
        if not filtered_input: 
            score = scores[teei, tori]
        else: 
            mentee = mentees.get(teei) # lowest avg matching to this mentee
            mentor = mentors.get(tori) # best mentor in mentee's row
            score = scores[teeindex, torindex]
        prple = prples[teei, tori]
        green = greens[teei, tori]
        yllow = yllows[teei, tori]
        matches.append([score, prple, green, yllow, teei, tori, mentee['NetID'], mentor['NetID']])
    
    return matches
    
#%% ANALYSIS

def analyze(matches, method, step):

    scoresonly = np.array([row[0] for row in matches])
    prplesonly = np.array([row[1] for row in matches])
    yllowsonly = np.array([row[2] for row in matches])
    greensonly = np.array([row[3] for row in matches])
    
    ors = sum(scoresonly)
    ops = (len(mentees) + 2) * totpts
    mnn = np.mean(scoresonly)
    mnp = np.mean(prplesonly)
    mng = np.mean(greensonly)
    mny = np.mean(yllowsonly)
    sdv = np.std(scoresonly)
    mdn = np.median(scoresonly)
    hgh = max(scoresonly)
    low = min(scoresonly)
    hgp = max(prplesonly)
    lop = min(prplesonly)
    hgg = max(greensonly)
    log = min(greensonly)
    hgy = max(yllowsonly)
    loy = min(yllowsonly)
    
    print('\nType of Run ~~~~~~~~~~~~~~~~ ' + method)
    print('Time of Run ~~~~~~~~~~~~~~~~ ' + str(dt.datetime.now()))
    #print('Scenarios Tested ~~~~~~~~~~~ ' + str(shuffleit))
    #print('Possble Scenarios ~~~~~~~~~~ ' + str(tst))
    #print('Proportion Tested ~~~~~~~~~~ ' + str(shuffleit/tst))
    print('Total Number of Matches ~~~~ ' + str(len(matches)))
    print('Remaining Mentees ~~~~~~~~~~ ' + str(len(mentees)-len(matches)) + '/' + str(len(mentees)))
    print('Remaining Mentors ~~~~~~~~~~ ' + str(len(mentors)-len(matches)) + '/' + str(len(mentors)))
    print('Overall Raw Score ~~~~~~~~~~ ' + str(ors))
    print('Overall Possible Score ~~~~~ ' + str(ops))
    print('Overall Match Quality ~~~~~~ ' + str(ors/ops*100) + '%')
    print('Mean Score ~~~~~~~~~~~~~~~~~ ' + str(mnn) + '/' + str(totpts) + ' = ' + str(mnn/totpts*100) + '%')
    print('Std. Dev. of Mean Score ~~~~ ' + str(sdv))
    print('Mean Purple Count ~~~~~~~~~~ ' + str(mnp/prppts) + '/~' + str(prpcnt))
    print('Mean Green Count ~~~~~~~~~~~ ' + str(mng/grnpts) + '/~' + str(grncnt))
    print('Mean Yellow Count ~~~~~~~~~~ ' + str(mny/ylwpts) + '/~' + str(ylwcnt))
    print('Median Score ~~~~~~~~~~~~~~~ ' + str(mdn) + '/~' + str(totpts))
    print('Best Match ~~~~~~~~~~~~~~~~~ ' + str(hgh) + '/~' + str(totpts))
    print('Worst Match ~~~~~~~~~~~~~~~~ ' + str(low) + '/~' + str(totpts))
    
    print('\nBest Purple Count ~~~~~~~~~~ ' + str(hgp/prppts) + '/~' + str(prpcnt))
    print('Worst Purple Count ~~~~~~~~~ ' + str(lop/prppts) + '/~' + str(prpcnt))
    print('Best Green Count ~~~~~~~~~~~ ' + str(hgg/grnpts) + '/~' + str(grncnt))
    print('Worst Green Count ~~~~~~~~~~ ' + str(log/grnpts) + '/~' + str(grncnt))
    print('Best Yellow Count ~~~~~~~~~~ ' + str(hgy/ylwpts) + '/~' + str(ylwcnt))
    print('Worst Yellow Count ~~~~~~~~~ ' + str(loy/ylwpts) + '/~' + str(ylwcnt))
    
    plt.figure(step, clear = True)
    plt.grid()
    plt.plot(np.arange(len(matches)), [row[0]*100/totpts for row in matches], 'o')
    plt.plot(np.linspace(0, len(matches), len(matches)), [mnn/totpts*100]*len(matches), '--')
    plt.title('BDB Match Progression (' + method + ')')
    plt.xlabel('Match Creation Time (Dec. Order of Mentee Match Difficulty)')
    plt.ylabel('Match Quality (%)')

#%% Matches to DataFrame
def matches_to_df(match_output, check = True, first_round = True): 
    d_keys = ['resp', 'first', 'last', 'email', 'school', 'study', 'clubs', 'programs', 'aff', 'hobbies', 'satnight', 'pres']
    if check: d_keys += ['alcohol', 'marijuana', 'relationship', 
                        'gender_id', 'ethnic_id', 'sexorient_id', 'firstgen_id', 'limitedinc_id', 'religion_id', 'dietary_id']
    pair_vars = ["_".join(["mentee", x]) for x in d_keys+['status', 'request_id','ability_id']]
    pair_vars += ["_".join(["mentor", x]) for x in d_keys + ['transfer OK?', 'gap-year']]
    satnight_phrases = {0: 'Reading a good book', 
                        1: 'Watching a movie with my best friend', 
                        2: 'Playing trivia at Krafthouse', 
                        3: 'Exploring the food scene in downtown Durham',
                        4: 'Wine night with some close friends',
                        5: 'Large party'}
    
    d_vars = {}
    for items in pair_vars: 
        if 'first' in items and 'firstgen' not in items: d_vars[items] = 'FirstName'
        elif 'last' in items: d_vars[items] = 'LastName'
        elif 'transfer OK?' in items: d_vars[items] = 'Q48'
        elif 'status' in items: d_vars[items] = 'Q3'
        elif 'gap-year' in items: d_vars[items] = 'Q58'
        elif 'email' in items: d_vars[items] = 'NetIDEmail'
        elif 'school' in items: d_vars[items] = 'Q11'
        elif 'study' in items: d_vars[items] = 'Q12'
        elif 'clubs' in items: d_vars[items] = 'Q15'
        elif 'programs' in items: d_vars[items] = 'Q18'
        elif 'aff' in items: d_vars[items] = 'Q17' if 'mentee' in items else 'Q16'
        elif 'hobbies' in items: d_vars[items] = 'Q23'
        elif 'satnight' in items: d_vars[items] = 'Q26' 
        elif 'pres' in items: d_vars[items] = 'Q31' if 'mentee' in items else 'Q49'
        elif 'alcohol' in items: d_vars[items] = 'Q21'
        elif 'marijuana' in items: d_vars[items] = 'Q22'
        elif 'relationship' in items: d_vars[items] = 'Q35'
        elif '_id' in items: 
            if 'gender' in items: d_vars[items] = 'Q36'
            elif 'ethnic' in items: d_vars[items] = 'Q37'
            elif 'sexorient' in items: d_vars[items] = 'Q38'
            elif 'firstgen' in items: d_vars[items] = 'Q39'
            elif 'limitedinc' in items: d_vars[items] = 'Q40' if 'mentee' in items else 'Q52'
            elif 'religion' in items: d_vars[items] = 'Q41' if 'mentee' in items else 'Q40'
            elif 'ability' in items: d_vars[items] = 'Q42'
            elif 'dietary' in items: d_vars[items] = 'Q43' if 'mentee' in items else 'Q42'
            elif 'request' in items: d_vars[items] = 'Q44'
        #else: print('{} not keyed'.format(items))
        
    d_pairs = {keys: [] for keys in pair_vars}
    for entries in match_output: 
        mentee_resp = mentees.get(entries[4])
        mentor_resp = mentors.get(entries[5])
        for items in pair_vars: 
            if 'mentee' in items: 
                if 'resp' in items: d_pairs[items] += [entries[4]]
                elif 'satnight' in items and not check: d_pairs[items] += [satnight_phrases[mentee_resp[d_vars[items]]]]
                elif 'study' in items: 
                    d_pairs[items] += [", ".join(list(filter(lambda x: x != 'None' and x != '', [mentee_resp[d_vars[items]], mentee_resp['Q13'], mentee_resp['Q14']]))).strip()]
                elif 'programs' in items: d_pairs[items] += [", ".join(list(filter(lambda x: x != 'None' and x != '',[mentee_resp[d_vars[items]], mentee_resp['Q16']]))).strip()]
                elif 'aff' in items: 
                    d_pairs[items] += [", ".join(list(filter(lambda x: ')' not in x, mentee_resp[d_vars[items]].split('(')))).strip()]
                elif 'status' in items: 
                    if 'first-year' in mentee_resp[d_vars[items]]: d_pairs[items] += ['First Year']
                    elif 'transfer' in mentee_resp[d_vars[items]]: d_pairs[items] += ['Transfer']
                    else: d_pairs[items] += ['Gap-Year']
                elif 'request' in items: 
                    if mentee_resp[d_vars[items]]!= 'None': d_pairs[items] += [mentee_resp[d_vars[items]].split(",")]
                    else: d_pairs[items] += [mentee_resp[d_vars[items]]]
                else: d_pairs[items] += [mentee_resp[d_vars[items]]]
            else: 
                if 'resp' in items: d_pairs[items] += [entries[5]]
                elif 'satnight' in items and not WARN: d_pairs[items] += [satnight_phrases[mentor_resp[d_vars[items]]]]
                elif 'study' in items: 
                    d_pairs[items] += [", ".join(list(filter(lambda x: x != 'None' and x != '', [mentor_resp[d_vars[items]], mentor_resp['Q13'], mentor_resp['Q14']]))).strip()]
                elif 'programs' in items: d_pairs[items] += [", ".join(list(filter(lambda x: x != 'None' and x != '',[mentor_resp[d_vars[items]], mentor_resp['Q17']]))).strip()]
                elif 'aff' in items: 
                    d_pairs[items] += [", ".join(list(filter(lambda x: ')' not in x, mentor_resp[d_vars[items]].split('(')))).strip()]
                else: d_pairs[items] += [mentor_resp[d_vars[items]]]
    df_pairs = pd.DataFrame.from_dict(d_pairs)
    df_pairs = df_pairs.set_index('mentee_resp')
    
    if check: 
        d_ids = {'Gender Identity': 'gender_id', 
         'Ethnicity/Race': 'ethnic_id', 
         'Dietary Restrictions': 'dietary_id', 
         'First Generation': 'firstgen_id', 
         'Limited Income': 'limitedinc_id', 
         'Religion': 'religion_id', 
         'Sexual Orientation': 'sexorient_id'} 
         #'Ability Status': 'ability_id'}

        transfer_warn = df_pairs.loc[(df_pairs['mentee_status'] == 'Transfer') & (df_pairs['mentor_transfer OK?'] == 'No')]
        if len(transfer_warn) != 0: print("{} transfers paired with mentors wanting first-years".format(len(transfer_warn)))
        gap_warn = df_pairs.loc[(df_pairs['mentee_status'] == 'Gap-Year') & (df_pairs['mentor_gap-year'] == 'No')]
        if len(gap_warn) != 0: print("{} gap-year students paired with mentors that did not undergo a gap-year".format(len(gap_warn)))
        
        df_alc_warn = df_pairs.loc[abs(df_pairs['mentor_alcohol']-df_pairs['mentee_alcohol']) > 1]
        df_marij_warn = df_pairs.loc[abs(df_pairs['mentor_marijuana']-df_pairs['mentee_marijuana']) > 1]
        df_satnight_warn = df_pairs.loc[abs(df_pairs['mentor_satnight']) - df_pairs['mentee_satnight'] > 1]
        df_relationship_warn = df_pairs.loc[abs(df_pairs['mentor_relationship']) - df_pairs['mentee_relationship'] > 1]
        
        df_ids_warn = pd.DataFrame(columns = df_pairs.columns)
        diff_ids = []
        for idx, row in df_pairs.iterrows(): 
            if row['mentee_request_id'] == 'None': continue
            row['mentee_request_id'] = list(filter(lambda x: x != 'None', row['mentee_request_id']))
            colnames = list(map(lambda x: d_ids.get(x), row['mentee_request_id']))
            colnames = list(filter(lambda x: pd.notna(x), colnames))
            mentee_ids = list(row.loc[list(map(lambda x: 'mentee_' + x, colnames))])
            mentor_ids = list(row.loc[list(map(lambda x: 'mentor_' + x, colnames))])
            if mentee_ids == mentor_ids: continue
            else: 
                row_diffs =[]
                for i in range(len(mentee_ids)):
                    if mentee_ids[i] != mentor_ids[i]: 
                        if (colnames[i] == 'ethnic_id' or colnames[i] == 'ability_id' or colnames[i] == 'dietary_id') and (mentee_ids[i] in mentor_ids[i] or mentor_ids[i] in mentee_ids[i]): 
                            if (len(mentor_ids[i]) != 0): continue
                            else: row_diffs += [row['mentee_request_id'][i]]
                        else: row_diffs += [row['mentee_request_id'][i]]
                if len(row_diffs) > 0: 
                    df_ids_warn= df_ids_warn.append(row)
                    diff_ids += [row_diffs]
        df_ids_warn['diff_ids'] = diff_ids
        df_ids_warn = df_ids_warn.loc[df_ids_warn['diff_ids'].apply(lambda x: len(x) >1)]
        
        diffs = set(list(df_alc_warn.index)+list(df_marij_warn.index)+list(df_satnight_warn.index)+list(df_relationship_warn.index) + list(df_ids_warn.index))
        diffs2 = list(diffs)
        
        df_fixed = pd.DataFrame(columns = df_pairs.columns)
        if not first_round: 
            fixed_idx = list(set(list(df_pairs.index)) - set(diffs2))
            for fix_i in fixed_idx:
                df_fixed = df_fixed.append(df_pairs.loc[fix_i])
            
        susscores = [] # [index, alc_diff, weed_diff, sat_diff, rel_diff, id_diff]
        for z in range(len(diffs2)):
            susscores.append([diffs2[z], 0, 0, 0, 0, 0])
            
        for z1 in range(len(df_alc_warn)):
            tmpd = abs(df_alc_warn.iloc[z1]['mentor_alcohol'] - df_alc_warn.iloc[z1]['mentee_alcohol'])
            tmpi = df_alc_warn.index[z1]
            susscores[diffs2.index(tmpi)][1] = tmpd
            #print("\nThis match (#{}) has an abs. mentor-mentee alc. diff. of {}/3".format(tmpi, tmpd))
            #print(matchesh[tmpi])
            
        for z2 in range(len(df_marij_warn)):
            tmpd = abs(df_marij_warn.iloc[z2]['mentor_marijuana'] - df_marij_warn.iloc[z2]['mentee_marijuana'])
            tmpi = df_marij_warn.index[z2]
            susscores[diffs2.index(tmpi)][2] = tmpd
            #print("\nThis match (#{}) has an abs. mentor-mentee marij. diff. of {}/3".format(tmpi, tmpd))
            #print(matchesh[tmpi])
    
        for z3 in range(len(df_satnight_warn)):
            tmpd = abs(df_satnight_warn.iloc[z3]['mentor_satnight'] - df_satnight_warn.iloc[z3]['mentee_satnight'])
            tmpi = df_satnight_warn.index[z3]
            susscores[diffs2.index(tmpi)][3] = tmpd
            #print("\nThis match (#{}) has an abs mentor-mentee sat. diff. of {}/5".format(tmpi, tmpd))
            #print(matchesh[tmpi])
    
        for z4 in range(len(df_relationship_warn)):
            tmpd = abs(df_relationship_warn.iloc[z4]['mentor_relationship'] - df_relationship_warn.iloc[z4]['mentee_relationship'])
            tmpi = df_relationship_warn.index[z4]
            susscores[diffs2.index(tmpi)][4] = tmpd
            #print("\nThis match (#{}) has an abs. mentor-mentee rel. diff. of {}/5".format(tmpi, tmpd))
            #print(matchesh[tmpi])
           
        for z5 in range(len(df_ids_warn)):
            tmpd = len(df_ids_warn.iloc[z5]['diff_ids'])
            tmpi = df_ids_warn.index[z5]
            susscores[diffs2.index(tmpi)][5] = tmpd
        
        print("\nv {}/{} pairings have at least one 2+ diff. (in alc/3, weed/3, sat/5, rel/5, id) v".format(len(diffs), len(df_pairs)))
        for z6 in range(len(susscores)):
            strikes = 5 - susscores[z6].count(0)
            if first_round: print("{:<60}{:>20} {:<4}".format(str(match_output[susscores[z6][0]-1]), str(susscores[z6]), 'X'*strikes))  
            else: 
                d_match = dict(zip(list(map(lambda x: x[4], match_output)), range(len(match_output))))
                print("{:<60}{:>20} {:<4}".format(str(match_output[d_match[susscores[z6][0]]]), str(susscores[z6]), 'X'*strikes))  
        print("X = one question with a 2+ difference")
        
        df_trouble = df_pairs.loc[diffs2]
        
        df_trouble['alc_diff'] = list(map(lambda x: x[1], susscores))
        df_trouble['weed_diff'] = list(map(lambda x: x[2], susscores))
        df_trouble['satnight_diff'] = list(map(lambda x: x[3], susscores))
        df_trouble['relationship_diff'] = list(map(lambda x: x[4], susscores))
        df_trouble['id_diff'] = list(map(lambda x: x[5], susscores))
    
    d_out = {'pairs': df_pairs, 
             'transfer_ck': transfer_warn, 
             'alc_ck': df_alc_warn, 
             'weed_ck': df_marij_warn, 
             'satnight_ck': df_marij_warn, 
             'relationship_ck': df_relationship_warn, 
             'ids_ck': df_ids_warn,
             'fixed_matches': df_fixed,
             'trouble_pairs': df_trouble}
    
    return d_out, diffs2, susscores

#%% DataFrame for Unmatched

def unmatched_to_df(responses, resp_ids, check = True, mentor = True):
    d_keys = ['resp', 'first', 'last', 'email', 'school', 'study', 'clubs', 'programs', 'aff', 'hobbies', 'satnight', 'pres']
    d_keys += ['alcohol', 'marijuana', 'relationship', 
               'gender_id', 'ethnic_id', 'sexorient_id', 'firstgen_id', 'limitedinc_id', 'religion_id', 'dietary_id']
    if mentor: solo_vars = ["_".join(["mentor", x]) for x in d_keys + ['transfer OK?', 'gap-year']]
    else: solo_vars = ["_".join(["mentee", x]) for x in d_keys+['status', 'request_id','ability_id']]
    satnight_phrases = {0: 'Reading a good book', 
                        1: 'Watching a movie with my best friend', 
                        2: 'Playing trivia at Krafthouse', 
                        3: 'Exploring the food scene in downtown Durham',
                        4: 'Wine night with some close friends',
                        5: 'Large party'}
    
    d_vars = {}
    for items in solo_vars: 
        if 'first' in items and 'firstgen' not in items: d_vars[items] = 'FirstName'
        elif 'last' in items: d_vars[items] = 'LastName'
        elif 'transfer OK?' in items: d_vars[items] = 'Q48'
        elif 'status' in items: d_vars[items] = 'Q3'
        elif 'gap-year' in items: d_vars[items] = 'Q58'
        elif 'email' in items: d_vars[items] = 'NetIDEmail'
        elif 'school' in items: d_vars[items] = 'Q11'
        elif 'study' in items: d_vars[items] = 'Q12'
        elif 'clubs' in items: d_vars[items] = 'Q15'
        elif 'programs' in items: d_vars[items] = 'Q18'
        elif 'aff' in items: d_vars[items] = 'Q17' if 'mentee' in items else 'Q16'
        elif 'hobbies' in items: d_vars[items] = 'Q23'
        elif 'satnight' in items: d_vars[items] = 'Q26' 
        elif 'pres' in items: d_vars[items] = 'Q31' if 'mentee' in items else 'Q49'
        elif 'alcohol' in items: d_vars[items] = 'Q21'
        elif 'marijuana' in items: d_vars[items] = 'Q22'
        elif 'relationship' in items: d_vars[items] = 'Q35'
        elif '_id' in items: 
            if 'gender' in items: d_vars[items] = 'Q36'
            elif 'ethnic' in items: d_vars[items] = 'Q37'
            elif 'sexorient' in items: d_vars[items] = 'Q38'
            elif 'firstgen' in items: d_vars[items] = 'Q39'
            elif 'limitedinc' in items: d_vars[items] = 'Q40' if 'mentee' in items else 'Q52'
            elif 'religion' in items: d_vars[items] = 'Q41' if 'mentee' in items else 'Q40'
            elif 'ability' in items: d_vars[items] = 'Q42' if 'mentee' in items else 'Q41'
            elif 'dietary' in items: d_vars[items] = 'Q43' if 'mentee' in items else 'Q42'
            elif 'request' in items: d_vars[items] = 'Q44'
        #else: print('{} not keyed'.format(items))
        
    d_solo = {keys: [] for keys in solo_vars}
    for resp in resp_ids: 
        student_resp = responses.get(resp)
        for items in solo_vars: 
            if 'resp' in items: d_solo[items] += [resp]
            elif 'satnight' in items and not check: d_solo[items] += [satnight_phrases[student_resp[d_vars[items]]]]
            elif 'study' in items: 
                d_solo[items] += [", ".join(list(filter(lambda x: x != 'None' and x != '', [student_resp[d_vars[items]], student_resp['Q13'], student_resp['Q14']]))).strip()]
            elif 'programs' in items: 
                if mentor: d_solo[items] += [", ".join(list(filter(lambda x: x != 'None' and x != '',[student_resp[d_vars[items]], student_resp['Q17']]))).strip()]
                else: d_solo[items] += [", ".join(list(filter(lambda x: x != 'None' and x != '',[student_resp[d_vars[items]], student_resp['Q18']]))).strip()]
            elif 'aff' in items: 
                d_solo[items] += [", ".join(list(filter(lambda x: ')' not in x, student_resp[d_vars[items]].split('(')))).strip()]
            else: d_solo[items] += [student_resp[d_vars[items]]]
    
    df_solo = pd.DataFrame.from_dict(d_solo)
    return df_solo

#%% CONCATENATE DATAFRAMES FOR SAVING
def save_df(d_out, prefix):
    df = pd.concat([d_out[prefix+"_P"], d_out[prefix+"_T"]])
    df.to_csv(prefix+'.txt', sep = '\t', index = False, header = True)
    return df

#%% CLEAN, SCORE, SHUFFLE, ANALYZE
d_out = {}
for mentees, mentors, school in [mentees_P, mentors_P, 'P'], [mentees_T, mentors_T, 'T']: 
    for x in range(len(mentees)):
        clean(mentees.get(x))
    for x in range(len(mentors)):
        clean(mentors.get(x))
    
    prppts = 60
    prpcnt = 11
    prptot = prppts * prpcnt
    grnpts = 30
    grncnt = 10 # note: approx b/c can have mult. majors in common
    grntot = grnpts * grncnt
    ylwpts = 15
    ylwcnt = 22 # note: approx b/c can have mult. interests in common
    ylwtot = ylwpts * ylwcnt
    totpts = grntot + ylwtot + prptot
    
    scores = np.zeros((len(mentees),len(mentors)))
    prples = np.zeros((len(mentees),len(mentors)))
    greens = np.zeros((len(mentees),len(mentors)))
    yllows = np.zeros((len(mentees),len(mentors)))
    for x in range(len(mentees)):
        for y in range(len(mentors)):
            compared = compare(mentees.get(x), mentors.get(y), prppts, grnpts, ylwpts)
            scores[x,y] = compared[0]
            prples[x,y] = compared[1]
            greens[x,y] = compared[2]
            yllows[x,y] = compared[3]
            
    scorebackup = scores
    
    matchesh = hunger(scores)
    analyze(matchesh, 'HUNGARIAN', 4)

#%% SAVE MATCHES

    SAVE_MATCH = True
    WARN = True # Should always be true - will break round 2 otherwise
    dfs_out, diffs2, susscores = matches_to_df(matchesh, check = WARN)
    
    df_pairs = dfs_out['pairs'].copy()
    df_transfer = dfs_out['transfer_ck']
    df_alc_warn = dfs_out['alc_ck']
    df_marij_warn = dfs_out['weed_ck']
    df_satnight_warn = dfs_out['satnight_ck']
    df_relationship_warn = dfs_out['relationship_ck']
    df_ids_warn = dfs_out['ids_ck']
        
    if SAVE_MATCH and WARN: 
        df_trouble_r1 = dfs_out['trouble_pairs']
        #df_trouble_r1 = df_trouble_r1.reset_index()
        d_out['trouble_pairs_r1_'+school] = df_trouble_r1
    
    matched_mentors_r1 = list(map(lambda x: int(x[5]), matchesh))
    unmatched_mentors_r1 = list(set(mentors.keys()) - set(matched_mentors_r1))
    d_tors = dict(zip(list(map(lambda x: x[4], matchesh)), list(map(lambda x: x[5], matchesh))))
    
    unmatched_mentees = list(set(mentees.keys()) - set(list(map(lambda x: int(x[4]), matchesh))))
    
    sustees = list(map(lambda x: x[0], susscores))
    sustors = list(map(lambda x: d_tors[x], sustees))
    
    new_matches = []
    tee_idx = dict(enumerate(sustees))
    unmatched_r1_cpy = unmatched_mentors_r1.copy() + sustors.copy()
    
    if len(unmatched_mentors_r1) == 0:
        df_unmatched = unmatched_to_df(mentees, unmatched_mentees, mentor = False)
        if SAVE_MATCH and WARN:
            d_out['trouble_pairs_r2_' + school] = pd.DataFrame()
    else: #can only try re-pairing process if more mentors than mentees
        for i in range(len(diffs2)):
            if diffs2[i] in unmatched_r1_cpy: unmatched_r1_cpy.remove(d_tors[diffs2[i]])
            print(unmatched_r1_cpy)
            scores_unmatched = scores[np.ix_(sustees, unmatched_r1_cpy)]
            tor_idx = dict(enumerate(list(unmatched_r1_cpy)))
            lst = hunger(scores_unmatched, filtered_input = True, tee_dict = tee_idx, tor_dict = tor_idx)
            print(len(lst))
            new_matches += [lst[i]]
            unmatched_r1_cpy += [d_tors[diffs2[i]]]
            unmatched_r1_cpy.remove(lst[i][5])
            print(lst[i][4], lst[i][5], len(unmatched_r1_cpy))
        
        dfs_out2, diffsr2, susscoresr2 = matches_to_df(new_matches, check = WARN, first_round = False)
        ## Automatically keeps first pairing if both rounds get flagged
        df_pairs.update(dfs_out2['fixed_matches'])
        
        ##Manual duplicate fixing:(reverting fixed ones to first round)
        df_duplicate_tors = df_pairs.loc[df_pairs.duplicated(subset = 'mentor_email', keep = False)]
        reverted = []
        for idx in df_duplicate_tors.index: 
            if all(df_pairs.loc[idx]['mentor_email'] != dfs_out['pairs'].loc[idx]):
                df_pairs.loc[idx] = dfs_out['pairs'].loc[idx]
                reverted += [idx]
        print("{} fixed pairs were reverted to first round: {}".format(len(reverted), reverted))
        
        df_trouble_r2 = dfs_out2['trouble_pairs']
        df_trouble_r2 = df_trouble_r2.append(df_trouble_r1.loc[df_trouble_r1['mentee_resp'].isin(reverted)])
        #df_trouble_r2 = df_trouble_r2.reset_index()
        if SAVE_MATCH and WARN: 
            d_out['trouble_pairs_r2_'+school] = df_trouble_r2
        
        matched_mentors = list(df_pairs['mentor_resp'])
        unmatched_mentors = list(set(mentors.keys()) - set(matched_mentors))
        
        df_pairs = df_pairs.reset_index()
        df_unmatched = unmatched_to_df(mentors, unmatched_mentors)
    
    d_out['pairs_'+school] = df_pairs
    if len(unmatched_mentees) == 0: 
        d_out['unmatched_tors_' + school] = df_unmatched
        d_out['unmatched_tees_' + school] = pd.DataFrame()
    else: 
        d_out['unmatched_tors_' + school] = pd.DataFrame()
        d_out['unmatched_tees_' + school] = df_unmatched
    
if SAVE_MATCH and WARN: 
    df_pairs = save_df(d_out, 'pairs')
    df_unmatched_tees = save_df(d_out, 'unmatched_tees')
    df_unmatched_tors = save_df(d_out, 'unmatched_tors')
    df_trouble_r1 = save_df(d_out, 'trouble_pairs_r1')
    df_trouble_r2 = save_df(d_out, 'trouble_pairs_r2')
#%% NOTES

'''

PACKAGE POINT QUINTUPLING AS BONUS POINTS

vvvv out of date vvv

output: match list (#s), totaled scores, mean score, median score, max score, min score, std. dev

1a. take random buddy, score their compatibility with all buddees, save the closest match
1b. (OR) take random buddy, match them with random buddee
2. repeat step 1 until all buddys used up
3. repeat steps 1 and 2 as many times as needed (simulation)
4. each line of array is like described output above; can sort by any of those parameters (likely balance of totaled scores and mean score?)
                    
Winning pairing combo of any given program run is output to text file somewhere (hopefully the same text file, in which a new line can be added every time the program runs). Need function that takes output to text file back in and finds winner within it. Also need one that can take winning combo and convert it into a readable format in a csv file.

'''