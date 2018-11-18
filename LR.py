from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import cPickle as pickle
from sklearn.linear_model import LogisticRegression


def regression(X,Y):
    
    regr = linear_model.LinearRegression()
    # regr = linear_model.Lasso()
    # regr = linear_model.Ridge(alpha=.1)
    # regr = linear_model.BayesianRidge()
    # regr=LogisticRegression()  #slow!
    
#     raw_data='Road-Accident.csv'
#     df = pd.read_csv('Road-Accident.csv',encoding='utf-8')
    num_test=1000
#     num_train=total_rows-num_test
    num_train=len(Y)-num_test

    X_train=X[:num_train, :]
    y_train=Y[:num_train]
    '''add noise, increase dataset'''
    # X_train0=X[:num_train, :]
    # y_train0=Y[:num_train]
    # for i in range(10):
    #     X_train=np.concatenate((X_train,X_train0+10*np.random.rand(np.shape(X_train0)[0],np.shape(X_train0)[1])))
    #     y_train=np.concatenate((y_train,y_train0))
    X_test=X[num_train:, :]
    y_test=Y[num_train:]

    # X_test=X[:1000, :]
    # y_test=Y[:1000]
    index_test=[i for i in range(num_test)]
    
#     for i in range (num_train):
#         X_train_row=[]
#         for data_index in data_index_set:
#             X_train_row.append(df[data_index][i])
#         X_train.append(X_train_row)
# #         print(np.shape(X_train))
#         y_train.append([df[pred_index][i]])
#     for i in range (num_train, total_rows):
#         X_test_row=[]
#         for data_index in data_index_set:
#             X_test_row.append(df[data_index][i])
#         X_test.append(X_test_row)        
# #         X_test.append([df[data_index,data_index][i]])
#         y_test.append([df[pred_index][i]])

    regr.fit(X_train, y_train)

    # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
    #                  normalize=False)
    y_pred=regr.predict(X_test)
    return index_test, y_test, y_pred
    

    
    

# print(regr.coef_)
# np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)

if __name__ == "__main__":
    start=time.time()
    # x_file = open("x.p",'r')
    # y_file = open("y.p",'r')
    # X=pickle.load(x_file)
    # Y=pickle.load(y_file)
    x_file=pd.read_csv("input0.6.csv",header = None)
    y_file=pd.read_csv("output.csv",header = None)
    X=x_file.values[1:]
    Y=y_file.values
    print(np.shape(X), np.shape(Y))

#     index_set=np.array([ 'vehicle_reference', 'vehicle_type', 'towing_and_articulation', 'vehicle_manoeuvre', 'vehicle_location-restricted_lane', 'junction_location', 'skidding_and_overturning', 'hit_object_in_carriageway', 'vehicle_leaving_carriageway', 'hit_object_off_carriageway', '1st_point_of_impact', 'was_vehicle_left_hand_drive?', 'journey_purpose_of_driver', 'sex_of_driver', 'age_of_driver', 'age_band_of_driver', 'engine_capacity_(cc)', 'propulsion_code', 'age_of_vehicle','driver_imd_decile', 'driver_home_area_type'])
    # index_set=np.array([ 'vehicle_reference', 'vehicle_type', 'towing_and_articulation', 'vehicle_manoeuvre', 'vehicle_location-restricted_lane', 'junction_location', 'skidding_and_overturning', 'hit_object_in_carriageway', 'vehicle_leaving_carriageway', 'hit_object_off_carriageway', '1st_point_of_impact', 'was_vehicle_left_hand_drive?', 'journey_purpose_of_driver', 'sex_of_driver', 'age_of_driver', 'age_band_of_driver', 'engine_capacity_(cc)', 'propulsion_code', 'age_of_vehicle','driver_imd_decile', 'driver_home_area_type'])
    # print(len(index_set))
    
    #     , 'vehicle_imd_decile', 'NUmber_of_Casualities_unique_to_accident_index', 'No_of_Vehicles_involved_unique_to_accident_index', 'location_easting_osgr', 'location_northing_osgr','longitude', 'latitude', 'police_force', 'accident_severity','number_of_vehicles', 'number_of_casualties', 'date', 'day_of_week',  'local_authority_(highway)', '1st_road_class', '1st_road_number', 'road_type', 'speed_limit', 'junction_detail', 'junction_control', '2nd_road_class', '2nd_road_number', 'pedestrian_crossing-human_control', 'pedestrian_crossing-physical_facilities', 'light_conditions', 'weather_conditions', 'road_surface_conditions', 'special_conditions_at_site', 'carriageway_hazards', 'urban_or_rural_area', 'did_police_officer_attend_scene_of_accident', 'lsoa_of_accident_location', 'casualty_reference', 'casualty_class', 'sex_of_casualty', 'age_of_casualty', 'age_band_of_casualty', 'casualty_severity', 'pedestrian_location', 'pedestrian_movement', 'car_passenger', 'bus_or_coach_passenger', 'pedestrian_road_maintenance_worker', 'casualty_type', 'casualty_home_area_type', 'casualty_imd_decile'])

#     'local_authority_(district)',
    
    #     ,'propulsion_code','age_of_vehicle','location_easting_osgr','location_northing_osgr','longitude','latitude','driver_imd_decile','police_force','number_of_casualties','day_of_week','local_authority_(district)','local_authority_(highway)','1st_road_class','1st_road_number','road_type','speed_limit','junction_detail','junction_control','2nd_road_class','light_conditions','road_surface_conditions','urban_or_rural_area','lsoa_of_accident_location','casualty_reference','casualty_class','sex_of_casualty','age_of_casualty','age_band_of_casualty','casualty_type','casualty_imd_decile','weather_conditions','road_surface_conditions','police_officer_attend','pedestrian_crossing-human_control', 'pedestrian_movement'])
#     print(index_set[2])
#     regression(270, 'age_of_driver','number_of_casualties')  #285331
    test_index, test_y, pred_y=regression(X,Y)  #285331
    me = (np.abs(test_y - pred_y)).mean()
    print('me: ',me)
    plt.scatter(test_index, test_y,s=8, label='label')
    plt.scatter(test_index, pred_y,s=4,label='prediction')
    leg = plt.legend(loc=1)

    plt.savefig('LR.png')
    plt.show()

    end=time.time()
    print('Time: ', start-end)