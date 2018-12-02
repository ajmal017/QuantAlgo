from Program import *

# Columns in CandleData list is as
# [date, open, high, low, close, volume, average, barCount]
def main():
    SetupLogger()
    logging.debug("now is %s", datetime.datetime.now())
    logging.getLogger().setLevel(logging.ERROR)

    cmdLineParser = argparse.ArgumentParser("api tests")
    # cmdLineParser.add_option("-c", action="store_True", dest="use_cache", default = False, help = "use the cache")
    # cmdLineParser.add_option("-f", action="store", type="string", dest="file", default="", help="the input file")
    cmdLineParser.add_argument("-p", "--port", action="store", type=int,
                               dest="port", default=7497, help="The TCP port to use")
    cmdLineParser.add_argument("-C", "--global-cancel", action="store_true",
                               dest="global_cancel", default=False,
                               help="whether to trigger a globalCancel req")
    args = cmdLineParser.parse_args()
    print("Using args", args)
    logging.debug("Using args %s", args)
    # print(args)

    def USStockWithPrimaryExch(symbol):
        #! [stkcontractwithprimary]
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
        #Specify the Primary Exchange attribute to avoid contract ambiguity
        #(there is an ambiguity because there is also a MSFT contract with primary exchange = "AEB")
        contract.primaryExchange = "ISLAND"
        #! [stkcontractwithprimary]
        return contract

    # args for retrieving candle data
    # endQueryTime = datetime.datetime.today().strftime("%Y%m%d %H:%M:%S")
    endQueryTime = datetime.datetime(2018,11,30, 23,59,59).strftime("%Y%m%d %H:%M:%S")
    # datetime.timedelta(days=180)
    timeSpam = "1 W"
    candleSpam = "5 mins"
    symbol = "SQ"
    contract = USStockWithPrimaryExch(symbol)
    dataFileName = "HistorialData/"+symbol+"_"+timeSpam+"_to_"+endQueryTime[:8]+"_"+candleSpam+".txt"

    try:
        app = TestApp(0, contract=contract, endQueryTime=endQueryTime, timeSpam=timeSpam, candleSpam=candleSpam)
        if args.global_cancel:
            app.globalCancelOnly = True
        # ! [connect]
        app.connect("127.0.0.1", args.port, clientId=0)
        # ! [connect]
        print("serverVersion:%s connectionTime:%s" % (app.serverVersion(),
                                                      app.twsConnectionTime()))
        # ! [clientrun]
        app.run()
        # ! [clientrun]
    except:
        raise
    finally:
        app.dumpTestCoverageSituation()
        app.dumpReqAnsErrSituation()

        f = open(dataFileName, 'w')
        try:
            for item in app.candleData:
                f.write("%s\n" % item)
        finally:
            f.close()

if __name__ == "__main__":
    main()