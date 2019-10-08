import numpy as np
import random
from collections import Counter
import scipy.stats as ST
import pprint
import os
import json
import copy


# Load simulator configuration variables ##########################################
def load_config_file():
    config_file = "sim_config.json"
    conf = {}
    try:
        fp = open(config_file, 'rb')
        conf = json.load(fp)
    except Exception, e:
        print "Error parcing the configuration file - " + str(e)
        print "Exiting..."
    return conf


def load_config_variables(c):
    d = { # Default values
        "num_users": 500,
        "top_websites": 1000,
        "avg_visits": 138,
        "avg_ads": 20,
        "targeted_ads_per": 0.1,
        "targeting_probability": 0.5,
        "sim_num": 1,
        "results_dir": "results",
        "debug": False,
        "randomize_users": False
    }
    
    val_str = { # Variables name mapping for user readability
        "number_of_users": "num_users",
        "top_websites": "top_websites",
        "average_ads": "avg_ads",
        "average_visits_per_user": "avg_visits",
        "targeted_ads_percentage": "targeted_ads_per",
        "targeting_probability": "targeting_probability",
        "number_of_simulations": "sim_num",
        "results_dir": "results_dir",
        "debug": "debug",
        "randomize_users": "randomize_users"
    }
    
    int_vars = ["number_of_simulations", "top_websites", "number_of_users", "average_visits_per_user", "average_ads"]
    float_vars = ["targeted_ads_percentage", "targeting_probability"]
    bool_vars = ["debug", "randomize_users"]
    
    for v in val_str:
        if v in c:
            if v == "results_dir":
                if c[v]:
                    d[val_str[v]] = c[v]
                    print v + " = " + str(c[v])
                else:
                    print v + " = " + str(d[val_str[v]])
            elif v in bool_vars:
                if c[v] == "True":
                    d[val_str[v]] = True
                    print v + " = " + str(c[v])
                else:
                    print v + " = " + str(d[val_str[v]])
            elif v in int_vars:
                try:
                    ni = int(c[v])
                    if ni > 0:
                        d[val_str[v]] = ni
                        print v + " = " + str(ni)
                    else:
                        print "'" + v + "' error. Provide a positive integer. Default value loaded."
                except Exception, e:
                    print "'" + v + "' error. Default value loaded. Error: " + str(e)
            elif v in float_vars:
                try:
                    nf = float(c[v])
                    if nf <= 1.0 and nf >= 0.0:
                        d[val_str[v]] = nf
                        print v + " = " + str(nf)
                    else:
                        print "'" + v + "' error. Provide a float number (1.0 - 0.0). Default value loaded." 
                except Exception, e:
                    print "'" + v + "' error. Default value loaded. Error: " + str(e)
            # END - elif
        else:
            print "'" + v + "' is not defined. Default loaded: " + str(d[val_str[v]])
    return d


def create_results_dir(d_name, lvl):
    if lvl == 1:
        tmp = d_name.split('/')
        if len(tmp) > 1:
            print "Please provide one level directory for the results..."
            exit()
    try:
        if not os.path.exists(d_name):
            os.mkdir(d_name)
    except Exception, e:
        print "Results folder creation error. " + str(e)
# Load simulator configuration variables - END ####################################


# Websites and visits creation ####################################################
def zipf_dist(num_samples, max_val):
    x = np.arange(1, max_val+1)
    a = 1.15
    weights = x ** (-a)
    weights /= weights.sum()
    bounded_zipf = ST.rv_discrete(name='bounded_zipf', values=(x, weights))
    s = bounded_zipf.rvs(size=num_samples)
    return s, bounded_zipf


def create_websites_popularity_dist(num_samples, max_val):
    return zipf_dist(num_samples, max_val)[0]
# Websites and visits creation - END ##############################################


# Advertisements creation #########################################################
def constrained_random_dist(s_size, avg_val):
    max_val = (avg_val * 2) + 1
    return np.random.randint(1, max_val, s_size)


def create_ads_per_website(t_web_set, avg_ads_):
    t_web = len(t_web_set)
    s = constrained_random_dist(t_web, avg_ads_)
    res = {}
    total = 0
    for k, v in zip(t_web_set, s):
        res[k] = v
        total += v
    return res, total


def create_ads_popularity(s_size, max_val):
    return zipf_dist(s_size, max_val)[0]


def select_targeted_ads(total_ads, p):
    s = np.random.choice([0, 1], size=total_ads, p=[1-p, p])
    targeted_ads = np.nonzero(s)[0]
    return list(targeted_ads)
# Advertisements creation - END ###################################################


# Simulation support functions ####################################################
def assign_static_ads_to_websites(t_web, in_ads_, napw_, total, t_ads):
    web = list(t_web)    
    res = {}
    finished = set()
    T_W = set()
    T_A = set()
    assigned = 0
    web_id = 0
    print "Websites:", len(web), "Ads:", len(in_ads_)
    while assigned < total:
        for ad in in_ads_:
            w = web[web_id]
            while w in finished:
                web_id += 1
                if web_id >= len(web):
                    web_id = 0
                w = web[web_id]
                if len(finished) == len(web):
                    print 'Total ads selected:', len(T_A), 'Total websites:', len(T_W)
                    return res
    
            if w not in res:
                res[w] = []
            if len(res[w]) < napw_[w]:
                if ad not in res[w]:
                    res[w].append(ad)
                    web_id += 1
                    if web_id >= len(web):
                        web_id = 0
                    assigned += 1
                    T_A.add(ad)
                    T_W.add(w)
            else:
                finished.add(w)       
    print 'Total ads selected:', len(T_A), 'Total websites:', len(T_W)
    return res


def assign_number_of_users_per_tads(tt_ads, n_users):
    tt = {}
    s, d = zipf_dist(len(tt_ads), int(round(n_users*0.10)))
    for targeted_ad, number_of_users in zip(tt_ads, s):
        tt[targeted_ad] = number_of_users
    return tt


def select_total_targeted_ads_per_user(t_users_num, n_users):
    t_ads_per_u = {}
    print 'Total users:', n_users
    user = 0
    for t_ad in t_users_num:
        if user < n_users:
            for i in range(t_users_num[t_ad]):
                if user not in t_ads_per_u:
                    t_ads_per_u[user] = set()
                t_ads_per_u[user].add(t_ad)
                user += 1
                if user >= n_users:
                    user = 0
        else:
            r_user = np.random.randint(0, n_users-1)
            
            if r_user not in t_ads_per_u:
                t_ads_per_u[r_user] = set()
                
            t_ads_per_u[r_user].add(t_ad)
            
            while t_ad not in t_ads_per_u[r_user]:
                t_ads_per_u[r_user].add(t_ad)
                r_user = np.random.randint(0, n_users-1)
    
    for u in t_ads_per_u:
        t_ads_per_u[u] = list(t_ads_per_u[u])
    
    return t_ads_per_u


def flip_ads(oldA, newA, x):
    for i in range(x):
        oldA.pop()
    return list(set(oldA) | newA)


def create_file_name(file_type, conf, n):
    ran_users = "False"
    if conf["randomize_users"]:
        ran_users = "True"
    f_name = file_type + \
    "-nu_" + str(conf["num_users"]) + \
    "-topw_" + str(conf["top_websites"]) + \
    "-avgv_" + str(conf["avg_visits"]) + \
    "-avga_" + str(conf["avg_ads"]) + \
    "-tap_" + str(conf["targeted_ads_per"]) + \
    "-tprop_" + str(conf["targeting_probability"]) + \
    "-iter_" + str(n) + \
    "-ran_" + ran_users + \
    ".json"
    return f_name


def save_to_file(f_name, data):
    fp = open(f_name, "w")
    json.dump(data, fp)
# Simulation support functions - END ##############################################


def run():
    # Load configuration ##############################################################
    print "Configuration:\n" + "=" * 14
    conf_f = load_config_file()
    conf = load_config_variables(conf_f)
    print conf
    create_results_dir(conf["results_dir"], 1)

    create_results_dir(conf["results_dir"] + "/data", 2)

    for it in range(conf["sim_num"]):
        print "\nSimulation number: " + str(it+1) + "\n" + "=" * 20
        print "Initialization:\n" + "=" * 15

        total_visits = conf["num_users"] * conf["avg_visits"]

        # Create websites and total visits based on bounded zipf distribution sampling ####
        w_dist = create_websites_popularity_dist(total_visits, conf["top_websites"]) # Has duplicate websites
        total_web = len(set(w_dist))
        print 'Total websites:', total_web, '- Total visits:', len(w_dist)
        if conf["debug"]:
            print 'w_dist:', w_dist
            print Counter(w_dist)

        # Check if the configuration is actually valid ###################################
        #if total_web <= (conf["num_users"]*1.5):
        #    print "\nERROR - Total websites are less than the total number of users. " + \
        #    "\nIncrease the number of top websites. Simulation halted..."
        #    break

        # Create the number of ads per website ############################################
        napw, total_th_ads = create_ads_per_website(set(w_dist), conf["avg_ads"])
        print 'Total theoretical ads:', total_th_ads, "- Websites:", len(napw)
        if conf["debug"]:
            print napw # {website: number_of_ads, ...}

        ADs = create_ads_popularity(total_th_ads, total_th_ads) # Has duplicate ads
        print 'Total actual ads:', len(set(ADs)), "- With duplicates:", len(ADs)
        if conf["debug"]:
            print 'ADs:', ADs
            print Counter(ADs)

        T_ADs = select_targeted_ads(len(set(ADs)), conf["targeted_ads_per"]) # [ad_id1, ad_id2, ...]
        print 'Total targeted ads:', len(T_ADs)
        if conf["debug"]:
            print 'T_ADs:', T_ADs

        # Assign ads to websites ##########################################################
        st_ads_per_web = assign_static_ads_to_websites(set(w_dist), ADs, napw, total_th_ads*2, T_ADs) # {website_id: [ad1,ad2,...], ...}
        if conf["debug"]:
            print 'st_ads_per_web:'
            print st_ads_per_web, '\n'

        # Define number of users per targeted ad ########################################## 
        num_targeted_users_per_ad = assign_number_of_users_per_tads(set(T_ADs), conf["num_users"])
        if conf["debug"]:
            print 'num_targeted_users_per_ad:'
            print num_targeted_users_per_ad, '\n'

        # Select total targeted ads per user ##############################################
        targeted_ads_per_user = select_total_targeted_ads_per_user(num_targeted_users_per_ad, conf["num_users"])
        if conf["debug"]:
            print 'targeted_ads_per_user:', targeted_ads_per_user

        # Simulation main loop ############################################################
        print "\nStarting simulation..."

        RAW_REC = []
        REC = {}

        t_users_c = {} # Hold the number of targeted users per targeted ad
        t_user_ads = {}

        TT_st = set()
        TT_ta = set()

        sel_web = 0
        last_w = {}
        if not conf["randomize_users"]:
            user = 0

        while total_visits > 0:
            if conf["randomize_users"]:
                user = zipf_dist(1, conf["num_users"])[0][0]
                user = user - 1

            # Initialize the targeted ads of the user
            if user not in t_user_ads: 
                t_user_ads[user] = set()

            # Select a website
            website = w_dist[sel_web]

            # Static ads selection ############################################################
            # Get the ads for the specific website
            s_ads = copy.copy(st_ads_per_web[website])

            # Check if we have already visit this website in order to randomize the ads
            if website in last_w:

                # Last surved ads on the specific website
                s_ads = last_w[website]

                # Select the number of ads to change
                to_flip = len(s_ads)

                # Select the replacement ads
                new_ads = set()
                while len(new_ads) < to_flip:
                    ss = np.random.choice(ADs, size=1)[0]
                    if ss not in s_ads: # Prevent targeted ads to appear as static over time
                        new_ads.add(ss)

                comp = flip_ads(s_ads, new_ads, to_flip)

                # Save the new ads for the specific website
                last_w[website] = comp
            else:
                # The initial ads for the specific website
                last_w[website] = s_ads
            # Static ads selection - END ######################################################

            # Targeted ads selection ##########################################################
            web_max_t = 1
            if 1 < (len(s_ads)/2):
                web_max_t = np.random.randint(1, len(s_ads)/2)

            # Select some random targeted ads
            tmp_t_ads = set()
            if user in targeted_ads_per_user:
                if len(targeted_ads_per_user[user]) > web_max_t:
                    while len(tmp_t_ads) < web_max_t:
                        t_index = np.random.randint(1, len(targeted_ads_per_user[user]))
                        t_ad = targeted_ads_per_user[user][t_index]
                        tmp_t_ads.add(t_ad)
                else:
                    tmp_t_ads = targeted_ads_per_user[user]
            else:
                tmp_t_ads = []

            targetedAds = []
            staticAds = copy.copy(last_w[website])
            for ttad in tmp_t_ads:
                if np.random.random() < conf["targeting_probability"]:
                    targetedAds.append(ttad)
                    TT_ta.add(ttad)

            for i in range(len(targetedAds)):
                staticAds.pop()
            # Targeted ads selection - END ####################################################

            # Collect simulated data ##########################################################
            if user not in REC:
                REC[user] = []
            rec = {'website': website, 'static_ads': staticAds, 'targeted_ads': targetedAds}
            REC[user].append(rec)
            raw_rec = {'user': user, 'website': website, 'static_ads': staticAds, 'targeted_ads': targetedAds}
            RAW_REC.append(raw_rec)
            # Collect simulated data - END ####################################################

            # Move to the next website
            sel_web += 1
            if sel_web > len(w_dist)-1:
                sel_web = 0
            # Reduce the total visits by one
            total_visits -= 1
            if total_visits == 0:
                break
            if not conf["randomize_users"]:
                user += 1
                if user >= conf["num_users"]:
                    user = 0

            if conf["debug"]:
                print 'Remaining visits:', total_visits
        rec_fn = create_file_name("REC",conf, it+1)
        print "Saving results in ./" + conf["results_dir"] + "/data/" + rec_fn, " - Records size:", len(REC)
        save_to_file(conf["results_dir"] + "/data/" + rec_fn, REC)
        raw_rec_fn = create_file_name("RAWREC", conf, it+1)
        print "Saving results in ./" + conf["results_dir"] + "/data/" + raw_rec_fn, "- Records size:", len(RAW_REC)
        save_to_file(conf["results_dir"] + "/data/" + raw_rec_fn, RAW_REC)

    print "\nAll Simulation(s) finish. Results can be found in ./" + conf["results_dir"] + "/data/"

    
if __name__ == '__main__':
    run()