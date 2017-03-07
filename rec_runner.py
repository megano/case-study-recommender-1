import pandas as pd
import numpy as np
import graphlab as gl
from bs4 import BeautifulSoup
from pprint import pprint
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def add_joke_len(filename):
    r = open(filename)
    soup = BeautifulSoup(r, 'html.parser')

    joke_list = soup.find_all('p')

    joke_len_list = []
    q_list = []
    clinton_list = []

    for x in joke_list:
        if x.find('?') > 0:
            q = 1
        else:
            q = 0
        if x.find('Clinton') > 0:
            clinton = 1
        else:
            clinton = 0
        joke_len_list.append(np.log(len(x)))
        q_list.append(q)
        clinton_list.append(clinton)

    avg_similarity = np.array([ 0.02919092,  0.02759409,  0.00968532,  0.01788923,  0.00682931,
       0.02301323,  0.02429581,  0.01188107,  0.0218484 ,  0.01943399,
       0.01123828,  0.02359791,  0.01254333,  0.02569454,  0.01773403,
       0.00954287,  0.02407927,  0.01385469,  0.01505321,  0.01212943,
       0.01357792,  0.01969422,  0.01108395,  0.00712827,  0.01787093,
       0.01934299,  0.01720096,  0.02890396,  0.01769203,  0.01185635,
       0.01800185,  0.02769295,  0.01833857,  0.01579566,  0.01427675,
       0.02783549,  0.02195491,  0.02377229,  0.02587114,  0.01923327,
       0.01495474,  0.02095016,  0.01703944,  0.02078577,  0.02436381,
       0.02473399,  0.02067519,  0.01776745,  0.02422141,  0.02609303,
       0.01507603,  0.0093577 ,  0.02898229,  0.01656648,  0.01464982,
       0.02632864,  0.00855076,  0.0148265 ,  0.00938203,  0.0181303 ,
       0.02213521,  0.02004861,  0.0212308 ,  0.00749268,  0.02296577,
       0.01610889,  0.01393321,  0.04260604,  0.02617758,  0.01746225,
       0.01829584,  0.01529294,  0.01799566,  0.02242413,  0.01262908,
       0.03388975,  0.00666667,  0.01243958,  0.01025946,  0.02188953,
       0.02570635,  0.01515648,  0.01025857,  0.01040254,  0.02141484,
       0.01867199,  0.03175679,  0.01994185,  0.0118524 ,  0.0178506 ,
       0.01711281,  0.02184123,  0.0287066 ,  0.022685  ,  0.0094119 ,
       0.01146307,  0.01708327,  0.01328454,  0.01012345,  0.01179002,
       0.01242792,  0.01967614,  0.02329067,  0.01747185,  0.02044013,
       0.0336918 ,  0.0139204 ,  0.02706954,  0.02080146,  0.01493801,
       0.0165347 ,  0.01866338,  0.03047691,  0.01634416,  0.02406709,
       0.0339638 ,  0.02122898,  0.03436507,  0.03301692,  0.01861853,
       0.01405534,  0.01023803,  0.01629764,  0.01589835,  0.02001565,
       0.01207473,  0.03081609,  0.02408324,  0.01490476,  0.0297394 ,
       0.02216062,  0.01174015,  0.03073034,  0.02341649,  0.01462874,
       0.02623898,  0.0155256 ,  0.01611861,  0.01407783,  0.01112362,
       0.00934034,  0.02792675,  0.02102156,  0.02378993,  0.01199531,
       0.01604801,  0.02562478,  0.02435416,  0.01676706,  0.01666902])
    return q_list, joke_len_list, clinton_list, avg_similarity


if __name__ == "__main__":
    sample_sub_fname = "data/sample_submission.csv"
    ratings_data_fname = "data/ratings.dat"
    output_fname = "data/test_ratings.csv"
    test_set = "data/validation_data.csv"

    filename = 'data/jokes.dat'
    q_list, joke_len, clinton_list, avg_similarity = add_joke_len(filename)

    joke_info = gl.SFrame({'joke_id': range(1,151), 'joke_len': joke_len, 'question': q_list, 'sim': avg_similarity})
    #joke_info = gl.SFrame({'joke_id': range(1,151),  'question': q_list, 'sim': avg_similarity})

    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    sample_sub = pd.read_csv(sample_sub_fname)
    for_prediction = gl.SFrame(sample_sub)
    #rec_engine = gl.factorization_recommender.create(observation_data=ratings, user_id="user_id", item_id="joke_id", target='rating', item_data = joke_info, solver='auto', linear_regularization= 1e-09, max_iterations= 50, num_factors= 64, regularization= 0)
    rec_engine = gl.factorization_recommender.create(observation_data=ratings, user_id="user_id", item_id="joke_id", target='rating', item_data = joke_info, solver='auto', linear_regularization= 1e-09, max_iterations= 50, num_factors= 3, regularization= 1e-06)


    predictions = [min(max(x, -10),10) for x in rec_engine.predict(ratings)]
    rmse = np.sqrt(mean_squared_error(ratings['rating'], predictions))
    print 'train rmse: {}'.format(rmse)

    ratings_test = pd.read_csv(test_set)
    for_prediction_test = gl.SFrame(ratings_test)
    predictions_test = [min(max(x, -10),10) for x in rec_engine.predict(for_prediction_test)]
    test_rmse = np.sqrt(mean_squared_error(ratings_test['rating'], predictions_test))
    print 'test rmse: {}'.format(test_rmse)

    sample_sub.rating = rec_engine.predict(for_prediction)
    sample_sub.rating = [min(max(x, -10),10) for x in sample_sub.rating]
    sample_sub.to_csv(output_fname, index=False)

    '''
    kfolds = gl.cross_validation.KFold(ratings, 5)
    params = dict(user_id='user_id', item_id='joke_id', target='rating',
                  solver='auto', side_data_factorization=False)

    paramsearch = gl.model_parameter_search.create(
                        kfolds,
                        gl.recommender.factorization_recommender.create,
                        params)

    print "best params by rmse:"
    pprint(paramsearch.get_best_params('mean_validation_rmse'))
    '''
