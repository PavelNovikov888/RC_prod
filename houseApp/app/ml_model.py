import pandas as pd
import numpy as np
import pickle
import random
import scipy
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity

# PATH = '/home/user/Рабочий стол/Prod_f/houseApp/app/files/'
PATH = './app/files/'

def user_service(user):
  #   # Десериализация
  with open (f'{PATH}model_base', 'rb') as fp:
    model_base = pickle.load(fp)
  with open (f'{PATH}model_users', 'rb') as fp:
    model_users = pickle.load(fp)

  with open (f'{PATH}train_interactions', 'rb') as fp:
    train_interactions = pickle.load(fp)
  with open (f'{PATH}inv_item_mappings', 'rb') as fp:
    inv_item_mappings = pickle.load(fp)
  with open (f'{PATH}inv_user_mappings', 'rb') as fp:
    inv_user_mappings = pickle.load(fp)
  with open (f'{PATH}item_metadata_list', 'rb') as fp:
    item_metadata_list = pickle.load(fp)
  with open (f'{PATH}user_metadata', 'rb') as fp:
    user_metadata = pickle.load(fp)
  with open (f'{PATH}user_metadata_mappings_users', 'rb') as fp:
    user_metadata_mappings_users = pickle.load(fp)
  with open (f'{PATH}item_mappings', 'rb') as fp:
    item_mappings = pickle.load(fp)
  try: 
    if user in list(set(inv_user_mappings.keys())):
      # Создадим матрицу всех пользователей и элементов, чтобы получить для них прогнозы
      n_users, n_items = train_interactions.shape

      # Используем lightFM to create predictions for all users and all items
      scoring_user_ids = np.concatenate([np.full((n_items, ), i) for i in range(n_users)]) # повторим user ID для всех проб
      scoring_item_ids = np.concatenate([np.arange(n_items) for i in range(n_users)]) # повторим весь диапазон идентификаторов item IDs x количество user
      scores = model_base.predict(user_ids = scoring_user_ids,
                                            item_ids = scoring_item_ids)
      scores = scores.reshape(-1, n_items) # получим одну строку на каждого user
      # Получим информацию о предыдущих покупках для каждого user
      previous = np.array(train_interactions.todense())
      
      f1 = f'User: {user}'
      f2 = f'У user: {user} имеется история взаимодействий с items.\nРекомендации строятся на основе коллаборативной фильтрации'
      # 3 лучших прогнозов для каждого пользователя
      k=3
      top_3 = np.argsort(-scores, axis=1) [::, :k]
      # print("Предыдущие покупки данного user:", *[inv_item_mappings.get(key) for key in np.array(range(previous.shape[1]))[previous[user]>0]], sep="\n")
      previous_buy = [inv_item_mappings.get(key) for key in np.array(range(previous.shape[1]))[previous[user]>0]]
      f3 = f'Предыдущие покупки данного user: {previous_buy}'
      top_3_recomend = sorted(zip([inv_item_mappings.get(key) for key in top_3[user]], range(k)), key = lambda x: x[1])
      f4 = f'Top 3 recommendations: {top_3_recomend}'
      # Удалим ранее купленные items из прогнозов
      top_3_new  = np.argsort(-(scores-(previous*999999)), axis=1)[::, :k] # вычтем предыдущие покупки из прогнозов
      top_3_newrecom = sorted(zip([inv_item_mappings.get(key) for key in top_3_new[user]], range(k)), key = lambda x: x[1])
      f5 = f'Top 3 recomedations без учета прошлых покупок: {top_3_newrecom}'
      return f1,f2,f3,f4,f5
    else:
      f1 = f'User: {user}'
      f2 = f'По данному user нет данных о взаимодействии с items.\nРекомендации строятся на основе матричного разложения'
      new_user_attriutes = random.sample(list(user_metadata),k=5)
      f3 = f'Для построения рекомендаций сгенерируем случайные признаки user: \n {new_user_attriutes}'
      user_indexes = [user_metadata_mappings_users.get(key) for key in new_user_attriutes]
      # Can either just weight each attribute equally
      weights = 1/len(user_indexes) # weight each metadata equally
      std_weights = [[weights] * len(new_user_attriutes)]

      # Combine the indexes we want populating with their weights
      new_user = np.zeros(len(user_metadata_mappings_users)) # create an empty array that will server as our dummy cold-user row
      np.put(new_user, user_indexes, std_weights) # update the relevant metadata attributes with the desired weights

      #  Now we can predict on this cold-user just like any other
      cold_user_preds = model_users.predict(user_ids = 0,
                                      item_ids = [*item_mappings.values()],
                                      item_features = item_metadata_list,
                                      user_features = scipy.sparse.csr_matrix(new_user))

      cold_ranks = np.argsort(-cold_user_preds)[:3]
      cold_ranks = pd.DataFrame(zip([*inv_item_mappings.values()], cold_ranks), columns = ["product_name", "rank"])
      f4 = 'Рекомендации items, сформированные на основе похожести признаков users \n'
      rec = list(cold_ranks.sort_values(["rank"])[:3]['product_name'].reset_index(drop=True))
      f5 = f'{rec}'
      return f1,f2,f3,f4,f5
  except:
    f1,f2,f3,f4,f5 = 1,2,3,4,5
    return f1,f2,f3,f4,f5
  
def item_service(itemid):
  #  # Десериализация
  with open (f'{PATH}model_base', 'rb') as fp:
    model_base = pickle.load(fp)
  with open (f'{PATH}train_interactions', 'rb') as fp:
    train_interactions = pickle.load(fp)
  with open (f'{PATH}inv_user_mappings', 'rb') as fp:
    inv_user_mappings = pickle.load(fp)

  try:
    # Создадим матрицу всех пользователей и элементов, чтобы получить для них прогнозы
    n_users, n_items = train_interactions.shape
    scores = model_base.predict(np.arange(n_users), np.repeat(itemid, n_users))
    # Получаем индексы пользователей, отсортированных по их вероятності взаимодействия с продуктом
    top_users_indices = np.argsort(-scores)
    f1 = f'Item: {itemid}'
    f2 = f'По данному item имеется история взаимодействий с users.\nРекомендации строятся на основе коллаборативной фильтрации'
    # Получаем индексы пользователей, отсортированных по их вероятности взаимодействия с продуктом
    top_users_indices = np.argsort(-scores)
    top_users = [inv_user_mappings[idx] for idx in top_users_indices][:3]
    f3 = f'Наиболее подходящие пользователи для продукта {itemid}: '
    f4 = [user for user in top_users]
    f5 = '\n'
    return f1,f2,f3,f4,f5
  except:
    f1,f2,f3,f4,f5 = 1,2,3,4,5
    return f1,f2,f3,f4,f5
 
def item_service_unknown(itemid):
#  # Десериализация
  with open (f'{PATH}model_items', 'rb') as fp:
    model_items = pickle.load(fp)

  with open (f'{PATH}train_interactions', 'rb') as fp:
    train_interactions = pickle.load(fp)
  with open (f'{PATH}inv_item_mappings', 'rb') as fp:
    inv_item_mappings = pickle.load(fp)
  with open (f'{PATH}inv_user_mappings', 'rb') as fp:
    inv_user_mappings = pickle.load(fp)
  with open (f'{PATH}item_metadata', 'rb') as fp:
    item_metadata = pickle.load(fp)
  with open (f'{PATH}item_metadata_list', 'rb') as fp:
    item_metadata_list = pickle.load(fp)
  with open (f'{PATH}item_metadata_mappings_items', 'rb') as fp:
    item_metadata_mappings_items = pickle.load(fp)
    
  try:
    f1 = f'Item: {itemid}'
    f2  = f'По данному item нет данных о взаимодействии с users.\n Рекомендации строятся на основе матричной факторизации'
    new_item_attriutes = random.sample(list(item_metadata),k=5)
    f3 = f'Для построения рекомендаций сгенерируем случайные признаки item: {new_item_attriutes}'
    # Получим indexes for the feature combinations we want to return embeddings for
    new_item_indexes = [item_metadata_mappings_items.get(key) for key in new_item_attriutes]
    # Can just weight each attribute equally
    weights = 1/len(new_item_indexes) # weight each metadata equally
    std_weights = [[weights] * len(new_item_attriutes)]

    new_item = np.zeros(len(item_metadata_mappings_items)) # create an empty array that will serve as our dummy cold-user row
    np.put(new_item, new_item_indexes, std_weights) # update the relevant metadata attributes with the desired weights

    # Convert it into a sparse matrix
    cold_item_matrix = scipy.sparse.csr_matrix(new_item)

    # Use LightFM to convert the matrix into embeddings
    cold_item_bias, cold_item_embedding = model_items.get_item_representations(cold_item_matrix)
    item_biases, item_embeddings  = model_items.get_item_representations(features = item_metadata_list)

    # Находим похожие items
    item_item_cold = pd.DataFrame(cosine_similarity(cold_item_embedding, item_embeddings).T, columns=(["cosine"]))
    item_item_cold["item_name"]=item_item_cold.index.to_series().map(inv_item_mappings)
    f4 = f'Вначале найдем похожие items: '
    f5 = item_item_cold.sort_values(by="cosine", ascending=False)[:5].to_html()
      
    # Create all user and item matrix to get predictions for it
    n_users, n_items = train_interactions.shape

    # Force lightFM to create predictions for all users and all items
    scoring_user_ids = np.concatenate([np.full((n_items, ), i) for i in range(n_users)]) # repeat user ID for number of prods
    scoring_item_ids = np.concatenate([np.arange(n_items) for i in range(n_users)]) # repeat entire range of item IDs x number of user
    scores = model_items.predict(user_ids = scoring_user_ids,
                                          item_ids = scoring_item_ids)
    scores = scores.reshape(-1, n_items) # get 1 row per user
    recommendations = pd.DataFrame(scores)

    # Extract the user and item representations
    user_biases, user_embeddings  = model_items.get_user_representations()
    # Create prediction score for our 'new' item
    recommendations["cold_ranking"] = ((user_embeddings @ cold_item_embedding.T + cold_item_bias).T + user_biases).T
    # recommendations.rank(axis=1, ascending=False)  # Highest value gets ranked as 1 i.e. best rec
    cold_rankings = recommendations.rank(axis=1, ascending=False)[["cold_ranking"]]

    # Add on users
    cold_rankings["user_id"]=cold_rankings.index.to_series().map(inv_user_mappings)
    cold_items = cold_rankings.sort_values(by="cold_ranking")[:3].to_html()
    f6 = f'Затем определим users подходящих по предпочтениям для данного item\n'
    f7 = cold_items
    return f1,f2,f3,f4,f5,f6,f7  
  except:
    f1,f2,f3,f4,f5,f6,f7 = 1,2,3,4,5
    return f1,f2,f3,f4,f5,f6,f7