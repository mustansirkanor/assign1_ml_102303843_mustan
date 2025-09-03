df = pd.read_csv("AWCustomers.csv")

selected_features = ['CustomerID', 'Title', 'FirstName', 'MiddleName', 'LastName',
                     'Suffix', 'AddressLine1', 'AddressLine2', 'City', 'StateProvinceName',
                     'CountryRegionName', 'PostalCode', 'PhoneNumber', 'BirthDate', 'Education',
                     'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NumberCarsOwned',
                     'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'LastUpdated']

df_selected = df[selected_features]

# Part 2

df_selected['YearlyIncome'] = df_selected['YearlyIncome'].fillna(df_selected['YearlyIncome'].mean())
df_selected['Education'] = df_selected['Education'].fillna(df_selected['Education'].mode()[0])
df_selected['NumberChildrenAtHome'] = df_selected['NumberChildrenAtHome'].fillna(df_selected['NumberChildrenAtHome'].median())
df_selected['NumberCarsOwned'] = df_selected['NumberCarsOwned'].fillna(df_selected['NumberCarsOwned'].median())
df_selected['TotalChildren'] = df_selected['TotalChildren'].fillna(df_selected['TotalChildren'].median())

df_selected['BirthDate'] = pd.to_datetime(df_selected['BirthDate'])
df_selected['Age'] = (pd.Timestamp.today() - df_selected['BirthDate']).dt.days // 365

numeric_features = ['Age', 'YearlyIncome', 'NumberChildrenAtHome', 'NumberCarsOwned', 'TotalChildren']
scaler = MinMaxScaler()
df_selected[numeric_features] = scaler.fit_transform(df_selected[numeric_features])

df_selected['Age_binned'] = pd.cut(df_selected['Age'], bins=5, labels=False)
df_selected['Income_binned'] = pd.qcut(df_selected['YearlyIncome'], 5, labels=False)

df_final = pd.get_dummies(df_selected, columns=['Gender', 'MaritalStatus', 'StateProvinceName'], drop_first=True)

print(df_final.shape)


# Part 3

obj1 = df_final.iloc[0]
obj2 = df_final.iloc[1]

numeric_cols = df_final.select_dtypes(include=['float64','int64','uint8']).columns
obj1_num = obj1[numeric_cols].values
obj2_num = obj2[numeric_cols].values

binary_features = df_final.select_dtypes(include=['uint8']).columns
obj1_bin = obj1[binary_features].values
obj2_bin = obj2[binary_features].values

if len(obj1_bin) > 0:
    simple_match = np.sum(obj1_bin == obj2_bin) / len(obj1_bin)
else:
    simple_match = np.nan
if np.sum(obj1_bin + obj2_bin) == 0:
    jaccard = 0.0
else:
    jaccard = jaccard_score(obj1_bin, obj2_bin, zero_division=0)

cos_sim = cosine_similarity([obj1_num], [obj2_num])[0][0]

print("Simple Matching Similarity:", simple_match)
print("Jaccard Similarity:", jaccard)
print("Cosine Similarity:", cos_sim)


if 'CommuteDistance' in df_final.columns:
    distance_mapping = {
        'Less than 1 Mile': 1,
        '1-2 Miles': 2,
        '2-5 Miles': 3,
        '5-10 Miles': 4,
        '10+ Miles': 5
    }
    df_final['CommuteDistance_num'] = df_final['CommuteDistance'].map(distance_mapping)
    correlation = df_final['CommuteDistance_num'].corr(df_final['YearlyIncome'])
    print("Correlation between CommuteDistance and YearlyIncome:", correlation)
else:
    print("Column 'CommuteDistance' not found in dataset")