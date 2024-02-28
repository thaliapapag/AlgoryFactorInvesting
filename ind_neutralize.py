"""
Three subcategories of where we need industry for for IndNeutralize:
- sector, industry, subindustry

Data neutralized within alphas

48:
- delta(delay(close,1),1),250 by subindustry

58
- vwap by sector

59
- weird weighted vwap

63
- close by industry

67
- vwap by sector
- adv20 by subindustry

70
- close by industry

76
- low by sector

79
- weighted sum of close and open by sector

80
- weighted sum of open and high by industry

82
- volume by sector

87
- adv81 by industry

89
- vwap by industry

90
- adv40 by subindustry

91
- close by industry

93
- vwap by industry

97
- weighted sum of low and vwap by industry

100
- This one gets really bad.
- Calc based on rank, close, low, high, by subindustry
- Correlation, close, rank(adv20), rank, ts_argmin by subindustry
"""
