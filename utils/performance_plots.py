# Overall propensity distribution
plt.hist(y_prob[:,1], bins=100)
plt.title("Call Propensity Distrubution")
plt.xlabel("Propensity Score")
plt.ylabel("Customer Volume")
plt.show()


# ROC curve
fpr, tpr, threshold = roc_curve(y_test, y_prob[:,1])

# Then we can plot the FPR against the TPR
def plot_roc_curve(fpr, tpr, _auc=False, label=None):
    if _auc==True:
        auc_v = auc(fpr, tpr)
        label=label + ' ROC curve (area = %0.2f)' % auc_v
        plt.plot(fpr, tpr, linewidth=2, label=label)
    else:
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0,1, 0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')

plot_roc_curve(fpr, tpr,_auc=True ,label='XGB')
plt.show()



# Gain chart
def plot_gain(y_test, proba_predict):
    rcParams['figure.figsize'] = 10, 7
    npos = np.sum(y_test)       # Number of positive instances in test set
    index = np.argsort(proba_predict[:, 1])   # indices sorted according to their score
    index = index[::-1]         # invert the indices, with instances with the highest score first
    y_test = y_test.reset_index(drop=True)
    sort_pos = y_test[index]    # sort the class membership according to the indices
    cpos = np.cumsum(sort_pos)  # cumulated sum
    recall = 100.0*cpos/npos    # recall column
    num = y_test.shape[0]       # num of instances in the test set
    size = np.arange(start=1, stop=num+1, step=1)   # target size
    index2 = np.argsort(y_test)         # Wizard
    index2 = index2[::-1]
    sort_pos2 = y_test[index2]
    cpos2 = np.cumsum(sort_pos2)
    wiz = 100.0*cpos2/npos
    size = 100.0*size /num         #target size in percentage
    plt.title('Gain Curve')
    plt.xlabel('% of Customers Proactively Contacted')
    plt.ylabel('% of Inbound Callers')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.scatter(size, size, marker='.', color='blue', s=1, label = 'Random')    # random
    plt.scatter(size, recall, marker='.', color='red', s=1, label = 'Model')   # gain curve
    plt.scatter(size, wiz, marker='.', color='green', s=1, label = 'Wizard')    # wizard curve
    plt.legend(loc="lower right")
    plt.show()


plot_gain(y_test, y_prob)
