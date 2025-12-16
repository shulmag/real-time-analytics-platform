using System;
using System.Configuration;

namespace FixClientLite
{
    using QuickFix;
    using QuickFix.Transport;

    public class FixClient : IApplication
    {
        private static log4net.ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
        private const string CONFIG_FILE_PATH = "ConfigFilePath";
        protected const string SESSION_INFO_PATH = "SessionInfoPath";
        private const string CRACKER_TYPE = "CrackerType";
        private const string CRACKER_CONFIGURATION = "CrackerConfiguration";
        private FileStoreFactory storeFactory = null;
        private FileLogFactory logFactory = null;
        private SessionSettings settings = null;

        private SocketInitiator initiator = null;
        private ThreadedSocketAcceptor acceptor = null;
        private ICompanyMessageCracker messageCracker = null;

        public FixClient()
        {
            try
            {
                InitializeServer();
            }
            catch (Exception mbs)
            {
                logger.ErrorFormat("An error occured at server initialization. The error is {0}", mbs.ToString());
                throw mbs;
            }

        }

        private void InitializeServer()
        {
            settings = new SessionSettings(ConfigurationManager.AppSettings[CONFIG_FILE_PATH]);
            storeFactory = new FileStoreFactory(settings);
            logFactory = new FileLogFactory(settings);
            messageCracker = FactoryInitialize(ConfigurationManager.AppSettings[CRACKER_TYPE]);            
            
            if (settings.ToString().Contains("CONNECTIONTYPE=acceptor")) //find another method
            {
                acceptor = new ThreadedSocketAcceptor(this, storeFactory, settings, logFactory);
                acceptor.Start();
            }
            else
            {
                initiator = new SocketInitiator(this, storeFactory, settings, logFactory);
                initiator.Start();
            }
            
        }

        private ICompanyMessageCracker FactoryInitialize(string crackerType)
        {           
            if (string.Compare(crackerType, "DUMMY", true) == 0)
            {
                var d = new DummyReceiver();
                d.Initialize(string.Empty);
                return d;
            }
            else if (string.Compare(crackerType, "CSV", true) == 0)
            {
                var d = new CsvReceiver();
                d.Initialize(ConfigurationManager.AppSettings[CRACKER_CONFIGURATION]);
                return d;
            }
            else if (string.Compare(crackerType, "MSSQL_DB", true) == 0)
            {
                var d = new DbReceiver();
                d.Initialize(ConfigurationManager.AppSettings[CRACKER_CONFIGURATION]);
                return d;
            }
            else if (string.Compare(crackerType, "BIGQUERY_DB", true) == 0)
            {
                var d = new DbReceiverBigQuery();
                d.Initialize(ConfigurationManager.AppSettings["CredentialKeys_File"], ConfigurationManager.AppSettings["Project_Id"], ConfigurationManager.AppSettings["DataSet_Id"], ConfigurationManager.AppSettings["Table_Id"]);
                return d;
            }
            else
            {
                throw new Exception($"Unknown CrackerType. Provided value:{crackerType} \n . Please check app config on key: <<crackerType>>. Implemented crackers: DUMMY, CSV, DB");
            }
        }

        public void FromAdmin(Message message, SessionID value)
        {
            try
            {
                logger.Debug(string.Format("fromAdmin...{0}", message.ToString()));
            }
            catch (Exception ex)
            {
                logger.ErrorFormat("error on processing fromadmin message. {0}", ex.ToString());
            }
        }

        public void FromApp(Message message, SessionID value)
        {
            try
            {
                //logger.Debug(string.Format("fromApp...{0}", message.ToString()));
                messageCracker.CrackMessage(message, value);                
            }
            catch (Exception ex)
            {
                logger.ErrorFormat("Error on crack the message. The error is {0}", ex.ToString());
            }
        }

        public void OnCreate(SessionID value)
        {
            logger.Debug(string.Format("OnCreate session...{0}", value.ToString()));
        }

        public void OnLogon(SessionID value)
        {
            logger.Debug(string.Format("OnLogon...{0}", value.ToString()));
            try
            {
                messageCracker.OnLogon(value);
            }
            catch (Exception ex)
            {
                logger.ErrorFormat($"error on onLogon for session ={value}. Error : {ex.ToString()}.");
            }
        }

        public void OnLogout(SessionID value)
        {
            logger.Debug(string.Format("OnLogout...{0}", value.ToString()));
            try
            {
                messageCracker.OnLogout(value);
            }
            catch (Exception ex)
            {
                logger.ErrorFormat($"error on onLogout for session ={value}. Error : {ex.ToString()}.");
            }
        }

        public void ToAdmin(Message message, SessionID value)
        {
            logger.Debug(string.Format("ToAdmin...{0}", message.ToString()));
        }

        public void ToApp(Message message, SessionID value)
        {
            logger.Debug(string.Format("ToApp...{0}", message.ToString()));
        }
    }
}