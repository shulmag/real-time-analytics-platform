
namespace FixClientLite
{
    using System;
    using System.Collections.Generic;
    using System.Configuration;
    using System.Data;
    using System.Data.SqlClient;
    using System.Linq;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;

    using Newtonsoft.Json;

    using QuickFix;
    using QuickFix.Fields;
    using QuickFix.FIX44;
    
    using Message = QuickFix.Message;
    using System.Globalization;
    using System.IO;
    using FixCommon;


    /// <summary>The wellington receiver.</summary>
    public class CsvReceiver : MessageCracker, ICompanyMessageCracker
    {
        private log4net.ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

        /// <summary>
        /// The connection string  key.
        /// </summary>
        private string connectionString;

        /// <summary>
        /// The connection string  key.
        /// </summary>
        private string connectionStringKey;
        private string tracesConnectionStringKey;
        
        
        public string ConnectionStringKey
        {
            get
            {
                return connectionStringKey;
            }
            set
            {
                this.connectionStringKey = value;
            }
        }

        public void CrackMessage(Message message, SessionID sessionID)
        {
            //Console.WriteLine(message.ToString());
            logger.DebugFormat("FromApp...{0}", message.ToString());
            string msgType = message.Header.GetString(Tags.MsgType);
            
            if (msgType.Equals(MsgType.INDICATION_OF_INTEREST.ToString()))
            {
                Trace o = MappingHelper.GetTrace((IndicationOfInterest)message);
                using (var sw = new StreamWriter(tracesConnectionStringKey, append: true))
                {
                    sw.WriteLine(o.GetCsvLine());
                }
            }
            else 
            {
                logger.ErrorFormat("Unknown message. Unable to parse the message. Fix message type= {0}", msgType);
            }
        }

        

        public void RequestOfferings(SessionID sessionId)
        {
            throw new NotImplementedException();
        }

        public void OnLogon(SessionID sessionId)
        {

        }

        public void OnLogout(SessionID sessionId)
        {

        }

        public void SendMessage(Message fixMessage, SessionID sessionId)
        {
            throw new NotImplementedException();
        }

        public void Initialize(string stringToConfigureFrom)
        {
            try
            {
                if (stringToConfigureFrom.IndexOfAny(Path.GetInvalidPathChars()) != -1)
                {
                    throw new Exception($"Please check the CrackerConfiguration value from config file. Expected a valid File Path. CheckedValue={stringToConfigureFrom}");
                }
                
                tracesConnectionStringKey = Path.Combine(stringToConfigureFrom,"Traces.csv");
                if (!File.Exists(tracesConnectionStringKey))
                {
                    using (var sw = new StreamWriter(tracesConnectionStringKey, append: true))
                    {
                        sw.WriteLine(Trace.GetHeader());
                    }
                }

            }
            catch (Exception ex)
            {
                logger.Error("An error occurs on initialize cracker using CrackerConfiguration value from config file. Expected a valid File Path.");
                throw ex;
            }
        }

        public void ToApp(Message message)
        {
            logger.DebugFormat("ToApp...{0}", message.ToString());
        }

        public void FromAdmin(Message message)
        {
            logger.DebugFormat("FromAdmin...{0}", message.ToString());
        }

        public void ToAdmin(Message message)
        {
            logger.DebugFormat("ToAdmin...{0}", message.ToString());
        }
    }
}

