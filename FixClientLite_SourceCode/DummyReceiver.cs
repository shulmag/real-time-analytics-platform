
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
  

    /// <summary>The wellington receiver.</summary>
    public class DummyReceiver : MessageCracker, ICompanyMessageCracker
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

